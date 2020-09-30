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

using Vect = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
using MapVect = Eigen::Map<Vect>;
template <std::size_t N>
using VectMat = Eigen::Matrix<float, N, Eigen::Dynamic, Eigen::RowMajor>;

struct Echos {

	std::minstd_rand rd;

	Echos(uint32_t seed) : rd(seed) {}

	Vect poisson(uint32_t L, float s, uint32_t d){
		Vect v(L);
		for(int i = d; i < L+d; i++){
			float b2 = float(i)*i - float(d)*d;
			float a2 = b2 - d*d/4.0;
			float arg = b2 > a2 ? 0 : sqrt(1 - b2/a2);
			float lam = s*4*sqrt(a2)*std::comp_ellint_2(arg);
			v[i-d] = std::poisson_distribution<uint32_t>(lam)(rd);
		}
		return v;
	}

	Vect phase(const Vect& echos){
		auto L = echos.size();
		std::normal_distribution<float> dist(0,1);
		Vect::NullaryExpr n(L, [&rd](){return dist(gen);});
		return echos.sqrt()*n;
	}

	void env(uint32_t d, float p, Vect& v){
		auto L = v.size();
		for(int i = d; i < L+d; i++)
			v[i-d] /= pow(i,p);
	}

	void decay(float r, Vect& v){
		auto L = v.size();
		for(int i = 0; i < L; i++)
			v[i] *= exp(-r*i);
	}

	Vect reverb(uint32_t L, uint32_t d, float s, float p, float r){
		auto x = phase(poisson(L,s,d));
		env(d,p,x);
		decay(r,x);
		return x;
	}

};

struct params {
	uint32_t seed;
	float length;
	float m2_per_tree;
	float decay;
	float atten;
	float dist;
	float rate;
	uint32_t buffersize;

	bool operator==(const params& p){
		return seed == p.seed &&
		length == p.length &&
		m2_per_tree == p.m2_per_tree &&
		decay == p.decay &&
		atten == p.atten &&
		dist == p.dist &&
		buffersize == p.buffersize &&
		rate == p.rate;
	}

	bool operator!=(const params& p){
		return !(operator==(p));
	}

	VectMat<4> make_impulses() const {
		uint32_t d = uint32_t(dist/c_sound*rate);
		float s = 1/m2_per_tree*(c_sound/rate)*(c_sound/rate);
		float r = atten/10*log(10)/rate;
		uint32_t L = uint32_t(length*rate);
		float p = decay;


		VectMat<4> imp(4, L);
		for(int i = 0; i < 4; i++)
			imp.row(i) = Echos(seed).reverb(L,d,s,p,r);

		imp /= imp.cwiseAbs().maxCoeff();
		return imp;
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
		MapVect xl(port[p_left_in], N),
			xr(port[p_right_in], N),
			yl(port[p_left_out], N),
			yr(port[p_right_out], N);

		params pars2{
			.seed = uint32_t(*port[p_seed]),
			.length = *port[p_length],
			.m2_per_tree = *port[p_density],
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

			MapVect(proc->inpdata(0), N) = xl;
			MapVect(proc->inpdata(1), N) = xr;

			proc->process(false);

			MapVect o1(proc->outdata(0), N),
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
