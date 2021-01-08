#pragma once
#include <future>
#include <iostream>
#include <random>
#include <chrono>
#include <set>
#include <zita-convolver.h>
#include <Eigen/Core>

#include <lvtk/plugin.hpp>
#include <lvtk/ext/worker.hpp>
#include <reverb.peg>

using namespace std::chrono;

using Arr = Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
using Map = Eigen::Map<Arr>;
using Arr2d = Eigen::Array<float, 2, Eigen::Dynamic, Eigen::RowMajor>;
using Mat2d = Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>;

inline static std::mt19937 rd(12345);
inline static std::normal_distribution<float> dist;
inline static const std::size_t MAX_L = p_ports[p_length].max*192000;
inline static const Arr2d randn = Arr2d::NullaryExpr(2, MAX_L, [](){return dist(rd);});

Arr iir_lowpass(Arr imp, const Arr& alpha, std::size_t k) {
	for (std::size_t i = 0; i < k; i++) {
		float y = 0;
		imp = imp.binaryExpr(alpha, [&y](auto x, auto a){
			return y += (x - y)*a;
		});
	}
	return imp;
}

Arr2d reverb(uint32_t L, float d, float p, float r, float c, std::size_t k = 16) {
	using namespace Eigen;
	Arr t = Arr::LinSpaced(L, 1, 1 + (L-1)/d);
	Arr t2 = t.square();

	// Compute the lowpass coefficient as if it is an ideal gaussian filter
	Arr alpha = 2 / (1 + sqrt(1 + (4*c/k)*t));

	Arr2d imp(2, L);
	imp << iir_lowpass(randn.row(0).segment(0, L), alpha, k),
	       iir_lowpass(randn.row(1).segment(0, L), alpha, k);

	Arr env = sqrt(sqrt(t2-1)*(t2-0.5))*exp((-p-1)*log(t) - r*(t-1));
	imp.rowwise() *= env.matrix().normalized().array();
	return imp;
}

struct params {
	using Result = std::tuple<params, std::unique_ptr<Convproc>>;

	float length, decay, atten, dist, damp, rate;
	uint32_t buffersize;

	bool operator <=>(params const &) const = default;

	Arr2d make_impulses() const {
		uint32_t L(length*rate);

		static const float c_sound = 343, w0 = 1e3*2*M_PI, db_to_exp = log(10)/20;
		float d = dist/c_sound*rate,
			r = atten*db_to_exp,
			p = decay,
			c = 2*db_to_exp*damp*pow(rate/w0, 2);

		return reverb(L,d,p,r,c);
	}

	Result operator()() const {
		auto imp = make_impulses();
		auto proc = std::make_unique<Convproc>();
		auto const L = imp.cols();
		proc->set_options(Convproc::OPT_VECTOR_MODE);
		auto const error = proc->configure(
			2, // # in channels
			2, // # out channels
			L,
			buffersize, // buffer size (quantum)
			buffersize, // min partition
			Convproc::MAXPART, // max partition
			0.f); // density

		if (error) {
			std::cout << "Error creating processor" << std::endl;
			proc = nullptr;
		} else {
			for (std::size_t i = 0; i < imp.rows(); i++)
				proc->impdata_create(i, i, 1, imp.row(i).data(), 0, L);
			proc->start_process(0, 0);
			if(proc->state() == Convproc::ST_WAIT)
				proc->check_stop();
		}
		return std::make_tuple(*this, std::move(proc));
	}

};


struct Reverb : public lvtk::Plugin<Reverb, lvtk::Worker> {
	inline constexpr static const char* URI = p_uri;

	params pars;

	std::packaged_task<params::Result()> job;
	std::future<params::Result> new_proc;
	std::unique_ptr<Convproc> proc;
	std::set<std::size_t> allowed{64, 128, 256, 512, 1024, 2048, 4096, 8192};

	float* port[p_n_ports];

	float rate;
	Reverb(const lvtk::Args& args_) :
		Plugin(args_), rate(args_.sample_rate)
	{}

	void connect_port (uint32_t p, void* data) {
		port[p] = static_cast<float*>(data);
	}

	void run(uint32_t N){
		if (allowed.find(N) == allowed.end())
			return;

		params pars2{
			.length = *port[p_length],
			.decay = *port[p_decay],
			.atten = *port[p_atten],
			.dist = *port[p_dist],
			.damp = *port[p_damp],
			.rate = rate,
			.buffersize = N,
		};


		if (new_proc.valid()) {
			if (new_proc.wait_for(0s) == std::future_status::ready) {
				auto [new_pars, new_proc_ptr] = new_proc.get();
				if(new_proc_ptr) {
					std::thread([](auto &&){}, std::move(proc)).detach();
					proc = std::move(new_proc_ptr);
					pars = new_pars;
				}
			}
		} else if (!proc || pars != pars2) {
			job = std::packaged_task<params::Result()>{pars2};
			new_proc = job.get_future();
			schedule_work(1, "");  // Just notify the thread to do work
		}

		if (N != pars.buffersize)
			return;

		Mat2d x(2, N);
		x << Map(port[p_left_in], N),
		     Map(port[p_right_in], N);
		x = x.array().isFinite().select(x, 0);

		if (proc) {
			if(proc->state() == Convproc::ST_WAIT)
				proc->check_stop();

			float gain = std::pow(10,*port[p_gain]/20);
			float cross = std::pow(10,*port[p_cross]/20);

			Mat2d mix(2, 2);
			mix << gain, cross*gain,
			       cross*gain, gain;

			Arr2d mixed = (mix*x).array();

			Map(proc->inpdata(0), N) = mixed.row(0);
			Map(proc->inpdata(1), N) = mixed.row(1);

			proc->process(false);

			Mat2d y(2, N);
			y << Map(proc->outdata(0), N),
			     Map(proc->outdata(1), N);

			if (*port[p_dry] > p_ports[p_dry].min) {
				float dry = std::pow(10,*port[p_dry]/20);
				x = x*dry + y;
			} else {
				x.swap(y);
			}
		}


		Map(port[p_left_out], N) = x.row(0).array();
		Map(port[p_right_out], N) = x.row(1).array();
	}

	lvtk::WorkerStatus work(lvtk::WorkerRespond &, uint32_t, const void*) {
		if (job.valid())
			job();
		return LV2_WORKER_SUCCESS;
	}
};

inline static const lvtk::Descriptor<Reverb> descriptor(p_uri);
