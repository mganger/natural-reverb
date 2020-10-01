#pragma once
#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <cmath>
#include <optional>
#include <future>
#include <iostream>
#include <random>
#include <chrono>
#include <zita-convolver.h>
#include <Eigen/Core>

#include <lvtk/plugin.hpp>
#include <reverb.peg>

using namespace std::chrono;

inline static constexpr float c_sound = 343;
inline static constexpr uint32_t maxsize = 96000*20;

using Arr = Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
using Map = Eigen::Map<Arr>;
template <std::size_t N>
using Arr2d = Eigen::Array<float, N, Eigen::Dynamic, Eigen::RowMajor>;

struct Echos {

	std::minstd_rand rd;
	std::normal_distribution<float> dist;

	Echos(uint32_t seed) : rd(seed), dist(0, 1) {}

	Arr ellipse_radius(const Arr& idx_d, float d){
		Arr b2 = idx_d.abs2() - d*d;
		Arr a2 = b2 - d*d/4.0;
		Arr arg = (1 - b2/a2).max(0).min(1).sqrt();
		return a2.sqrt()*arg.unaryExpr([](auto a) {
			return std::comp_ellint_2(a);
		});
	}

	Arr2d<4> randn(std::size_t N) {
		return Arr2d<4>::NullaryExpr(4, N, [this](){return dist(rd);});
	}

	Arr2d<4> reverb(uint32_t L, float d, float p, float r){
		auto idx = Arr::LinSpaced(L, 0, L-1);
		Arr idx_d = idx + d;

		Arr rad = ellipse_radius(idx_d, d);
		Arr env = rad.sqrt()*(p*idx_d.log() - r*idx).exp();
		env *= 1/env.maxCoeff();
		Arr2d<4> echos = randn(L);

		return echos.rowwise()*env;
	}

};

struct params {
	float length;
	float decay;
	float atten;
	float dist;
	float rate;
	uint32_t buffersize;

	bool operator==(const params& p){
		return length == p.length &&
		decay == p.decay &&
		atten == p.atten &&
		dist == p.dist &&
		buffersize == p.buffersize &&
		rate == p.rate;
	}

	bool operator!=(const params& p){
		return !(operator==(p));
	}

	Arr2d<4> make_impulses() const {
		float d = dist/c_sound*rate;
		float r = atten/10*log(10)/rate;
		uint32_t L(length*rate);
		float p = decay;

		auto seed = std::hash<float>()(dist);
		return Echos(seed).reverb(L,d,p,r);
	}

	std::shared_ptr<Convproc> make_processor() const {
		auto proc = std::make_shared<Convproc>();
		auto imp = make_impulses();
		proc->set_options(0);
		proc->configure(
			2, // # in channels
			4, // # out channels
			maxsize,
			buffersize, // buffer size (quantum)
			buffersize, // min partition
			buffersize, // max partition
			0.f); // density

		auto L = imp.cols();
		proc->impdata_create(0,0,1,imp.row(0).data(),0,L);
		proc->impdata_create(0,1,1,imp.row(1).data(),0,L);
		proc->impdata_create(1,2,1,imp.row(2).data(),0,L);
		proc->impdata_create(1,3,1,imp.row(3).data(),0,L);
		return proc;
	}

};


struct Reverb : public lvtk::Plugin<Reverb> {
	inline constexpr static const char* URI = p_uri;

	params pars;

	std::optional<std::future<std::shared_ptr<Convproc> > > new_proc;
	std::shared_ptr<Convproc> proc;

	float* port[p_n_ports];

	float rate;
	Reverb(const lvtk::Args& args_) :
		Plugin(args_), rate(args_.sample_rate)
	{}

	void connect_port (uint32_t p, void* data) {
		port[p] = static_cast<float*>(data);
	}

	void run(uint32_t N){
		Map xl(port[p_left_in], N),
			xr(port[p_right_in], N),
			yl(port[p_left_out], N),
			yr(port[p_right_out], N);

		params pars2{
			.length = *port[p_length],
			.decay = *port[p_decay],
			.atten = *port[p_atten],
			.dist = *port[p_dist],
			.rate = rate,
			.buffersize = N,
		};


		if (pars != pars2 && !new_proc)
			new_proc.emplace(std::async(std::launch::async, [pars2](){
				return pars2.make_processor();
			}));

		if (new_proc) {
			if (new_proc->wait_for(0s) == std::future_status::ready) {
				proc = new_proc->get();
				proc->start_process(0, 0);
				new_proc.reset();
			}
		}

		if (proc) {
			if(proc->state() == Convproc::ST_WAIT)
				proc->check_stop();

			Map(proc->inpdata(0), N) = xl;
			Map(proc->inpdata(1), N) = xr;

			proc->process(false);

			Map o1(proc->outdata(0), N),
				o2(proc->outdata(1), N),
				o3(proc->outdata(2), N),
				o4(proc->outdata(3), N);

			float gain = std::pow(10,*port[p_gain]/10);
			float cross = std::pow(10,*port[p_cross]/10);


			yl = o1*gain + o3*cross;
			yr = o2*gain + o4*cross;
		} else {
			yl = xl;
			yr = xr;
		}
	}
};

inline static const lvtk::Descriptor<Reverb> descriptor(p_uri);
