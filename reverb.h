#pragma once
#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <cmath>
#include <vector>
#include <optional>
#include <future>
#include <iostream>
#include <random>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <thread>
#include <zita-convolver.h>
#include <Eigen/Core>

#include <lvtk/plugin.hpp>
#include "reverb.peg"

using namespace std::chrono;

inline static constexpr double c_sound = 343;
inline static constexpr uint32_t maxsize = 96000*20;

using VectType = Eigen::VectorXf;

template<class Eng>
std::vector<uint32_t> poisson(uint32_t L, double s, uint32_t d, Eng& rd){
	std::cout << "poisson" << std::endl;
	std::vector<uint32_t> v(L);
	for(int i = d; i < L+d; i++){
		double b2 = double(i)*i - double(d)*d;
		double a2 = b2 - d*d/4.0;
		double arg = b2 > a2 ? 0 : sqrt(1 - b2/a2);
		double lam = s*4*sqrt(a2)*std::comp_ellint_2(arg);
		v[i-d] = std::poisson_distribution<uint32_t>(lam)(rd);
	}
	return v;
}

template<class Eng>
VectType phase(const std::vector<uint32_t>& echos, Eng& rd){
	std::cout << "phase" << std::endl;
	auto L = echos.size();
	std::cout << L << std::endl;
	VectType v(L);
	std::normal_distribution<double> dist(0,1);
	for(int i = 0; i < L; i++)
		v[i] = dist(rd)*sqrt(echos[i]);
	return v;
}

template <class Vect>
void env(uint32_t d, double p, Vect& v){
	std::cout << "env" << std::endl;
	auto L = v.size();
	for(int i = d; i < L+d; i++)
		v[i-d] /= pow(i,p);
}

template <class Vect>
void decay(double r, Vect& v){
	std::cout << "decay" << std::endl;
	auto L = v.size();
	std::cout << "r " << r << std::endl;
	for(int i = 0; i < L; i++)
		v[i] *= exp(-r*i);
}

template<class Eng>
VectType reverb(uint32_t L, uint32_t d, double s, double p, double r, Eng& rd){
	std::cout << "reverb" << std::endl;
	auto x = phase(poisson(L,s,d,rd),rd);
	env(d,p,x);
	decay(r,x);
	return x;
}

template <class T>
void copy(T* v1, T* v2, T a, uint32_t N){
	for(int i = 0; i < N; i++)
		v2[i] = v1[i]*a;
}

template <class T>
void mix(T* v1, T* v2, T a, uint32_t N){
	for(int i = 0; i < N; i++)
		v1[i] += v2[i]*a;
}

struct params {
	uint32_t seed;
	double length;
	double m2_per_tree;
	double decay;
	double atten;
	double dist;
	double rate;
	int buffersize;

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

	std::array<VectType, 4> make_impulses() const {
		uint32_t d = uint32_t(dist/c_sound*rate);
		double s = 1/m2_per_tree*(c_sound/rate)*(c_sound/rate);
		double r = atten/10*log(10)/rate;
		uint32_t L = uint32_t(length*rate);
		double p = decay;

		std::minstd_rand dev(seed);

		float max{0.0};
		std::array<VectType, 4> imp;

		for(int i = 0; i < 4; i++){
			imp[i] = reverb(L,d,s,p,r,dev);
			max = std::max(max, imp[i].cwiseAbs().maxCoeff());
		}

		std::cout << "max " << max << std::endl;

		for(auto& imp_i : imp)
			imp_i /= max;
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

		auto L = imp[0].size();
		proc->impdata_create(0,0,1,imp[0].data(),0,L);
		proc->impdata_create(0,1,1,imp[1].data(),0,L);
		proc->impdata_create(1,2,1,imp[2].data(),0,L);
		proc->impdata_create(1,3,1,imp[3].data(),0,L);
		return proc;
	}

};


struct Reverb : public lvtk::Plugin<Reverb> {
	inline constexpr static const char* URI = p_uri;

	params pars;

	std::array<VectType, 4> imp;
	std::optional<std::future<std::shared_ptr<Convproc> > > new_proc;

	std::shared_ptr<Convproc> proc;

	float* port[p_n_ports];

	double rate;
	Reverb(const lvtk::Args& args_) :
		Plugin(args_), rate(args_.sample_rate)
	{}

	void connect_port (uint32_t p, void* data) {
		port[p] = static_cast<float*>(data);
	}


	void run(uint32_t N){
		auto xl = port[p_left_in];
		auto xr = port[p_right_in];
		auto yl = port[p_left_out];
		auto yr = port[p_right_out];

		params pars2;
		pars2.seed = uint32_t(*port[p_seed]);
		pars2.length = *port[p_length];
		pars2.m2_per_tree = *port[p_density];
		pars2.decay = *port[p_decay];
		pars2.atten = *port[p_atten];
		pars2.dist = *port[p_dist];
		pars2.buffersize = N;
		pars2.rate = rate;


		if (pars != pars2)
			new_proc.emplace(std::async(std::launch::async, [pars2](){
				return pars2.make_processor();
			}));

		if (new_proc) {
			if (new_proc->wait_for(0s) == std::future_status::ready) {
				proc = new_proc->get();
				proc->start_process(0, 0);
			}
		}

		if(proc->state() == Convproc::ST_WAIT)
			proc->check_stop();

		std::copy(xl, xl+N, proc->inpdata(0));
		std::copy(xr, xr+N, proc->inpdata(1));

		proc->process(false);

		float* out[4];
		for(int i = 0; i < 4; i++)
			out[i] = proc->outdata(i);

		float gain = std::pow(10,*port[p_gain]/10);
		float cross = std::pow(10,*port[p_cross]/10);

		copy(out[0],yl,gain,N);
		copy(out[2],yr,gain,N);

		mix(yl,out[3],cross,N);
		mix(yr,out[1],cross,N);
	}
};

inline static const lvtk::Descriptor<Reverb> descriptor(p_uri);
