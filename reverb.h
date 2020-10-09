#pragma once
#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <cmath>
#include <future>
#include <iostream>
#include <random>
#include <chrono>
#include <set>
#include <zita-convolver.h>
#include <Eigen/Core>

#include <lvtk/plugin.hpp>
#include <reverb.peg>

using namespace std::chrono;

inline static constexpr float c_sound = 343;

using Arr = Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
using Map = Eigen::Map<Arr>;
template <std::size_t N>
using Arr2d = Eigen::Array<float, N, Eigen::Dynamic, Eigen::RowMajor>;
template <std::size_t N>
using Mat2d = Eigen::Matrix<float, N, Eigen::Dynamic, Eigen::RowMajor>;

inline static std::minstd_rand rd(12345);
inline static std::normal_distribution<float> dist;
inline static const std::size_t MAX_L = p_ports[p_length].max*192000;
inline static const Arr2d<2> randn = Arr2d<2>::NullaryExpr(2, MAX_L, [](){return dist(rd);});

template <class T>
Arr ellipse_integral(const T& t, float d){
	Arr b2 = t*(t + d);
	Arr a2 = (b2 - d*d/4.0).max(1e-30);
	Arr arg = (1 - b2/a2).max(0).min(1).sqrt();
	return a2.sqrt()*arg.unaryExpr([](auto a) {
		return std::comp_ellint_2(a);
	});
}

Arr2d<2> reverb(uint32_t L, float d, float p, float r){
	auto t = Arr::LinSpaced(L, 0, L-1);
	Arr rad = ellipse_integral(t, d);
	Arr env = rad.sqrt()*(-p*(t+d).log() - r*t).exp();
	Arr2d<2> imp = randn.block(0, 0, 2, L).rowwise()*env;
	return imp.matrix().rowwise().normalized().array();
}

struct params {
	using Result = std::tuple<params, std::unique_ptr<Convproc>>;

	float length;
	float decay;
	float atten;
	float dist;
	float rate;
	uint32_t buffersize;

	bool operator!=(const params& p){
		return length != p.length ||
		decay != p.decay ||
		atten != p.atten ||
		dist != p.dist ||
		buffersize != p.buffersize ||
		rate != p.rate;
	}

	Arr2d<2> make_impulses() const {
		float d = dist/c_sound*rate;
		float r = atten/20*log(10)/rate;
		uint32_t L(length*rate);
		float p = decay;

		return reverb(L,d,p,r);
	}

	Result operator()() const {
		auto imp = make_impulses();

		auto proc = std::make_unique<Convproc>();
		auto L = imp.cols();
		proc->set_options(Convproc::OPT_VECTOR_MODE);
		auto result = proc->configure(
			2, // # in channels
			2, // # out channels
			L,
			buffersize, // buffer size (quantum)
			buffersize, // min partition
			Convproc::MAXPART, // max partition
			0.f); // density

		if (result) {
			std::cout << "Error creating processor" << std::endl;
			proc = nullptr;
		} else {
			for (auto i : {0, 1})
				proc->impdata_create(i, i, 1, imp.row(i).data(), 0, L);
		}

		proc->start_process(0, 0);
		return std::make_tuple(*this, std::move(proc));
	}

};


struct Reverb : public lvtk::Plugin<Reverb> {
	inline constexpr static const char* URI = p_uri;

	params pars;

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
			.rate = rate,
			.buffersize = N,
		};


		if (new_proc.valid()) {
			if (new_proc.wait_for(0s) == std::future_status::ready) {
				auto [new_pars, new_proc_ptr] = new_proc.get();
				if(new_proc_ptr) {
					proc = std::move(new_proc_ptr);
					pars = new_pars;
				}
			}
		} else if (!proc || pars != pars2) {
			new_proc = std::async(std::launch::async, pars2);
		}

		if (N != pars.buffersize)
			return;

		Mat2d<2> x(2, N);
		x << Map(port[p_left_in], N),
		     Map(port[p_right_in], N);
		x = x.array().isFinite().select(x, 0);

		if (proc) {
			if(proc->state() == Convproc::ST_WAIT)
				proc->check_stop();

			float gain = std::pow(10,*port[p_gain]/20);
			float cross = std::pow(10,*port[p_cross]/20);

			Mat2d<2> mix(2, 2);
			mix << gain, cross*gain,
			       cross*gain, gain;

			Arr2d<2> mixed = (mix*x).array();

			Map(proc->inpdata(0), N) = mixed.row(0);
			Map(proc->inpdata(1), N) = mixed.row(1);

			proc->process(false);

			x << Map(proc->outdata(0), N),
			     Map(proc->outdata(1), N);
		}

		Map(port[p_left_out], N) = x.row(0).array();
		Map(port[p_right_out], N) = x.row(1).array();
	}
};

inline static const lvtk::Descriptor<Reverb> descriptor(p_uri);
