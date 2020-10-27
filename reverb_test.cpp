#define BOOST_TEST_MODULE reverb_tests
#include <boost/test/unit_test.hpp>
#include <reverb.h>

namespace utf = boost::unit_test::framework;

struct Fixture {
	static const int N = 256;
	float input[2][N];
	float output[2][N];
	float length = 3;
	float decay = 2.1;
	float atten = 0.1;
	float dist = 15;
	float rate = 96000;
	float gain = -1;
	float cross = -10;
	float dry = -10;
	lvtk::Args args;
	Reverb plugin;

	Fixture() : args(rate, "", {}), plugin(args) {
		plugin.connect_port(p_left_in, input[0]);
		plugin.connect_port(p_right_in, input[1]);
		plugin.connect_port(p_left_out, output[0]);
		plugin.connect_port(p_right_out, output[1]);
		plugin.connect_port(p_length, &length);
		plugin.connect_port(p_decay, &decay);
		plugin.connect_port(p_atten, &atten);
		plugin.connect_port(p_dist, &dist);
		plugin.connect_port(p_gain, &gain);
		plugin.connect_port(p_cross, &cross);
		plugin.connect_port(p_dry, &dry);
		BOOST_TEST_CHECKPOINT("Starting");
	}
};

BOOST_FIXTURE_TEST_SUITE(reverb_tests, Fixture)

BOOST_AUTO_TEST_CASE(simple) {
	BOOST_CHECK(!plugin.proc);
	BOOST_CHECK(!plugin.new_proc.valid());
	plugin.run(N);
	BOOST_CHECK(!plugin.proc);
	BOOST_CHECK(plugin.new_proc.valid());
}

BOOST_AUTO_TEST_CASE(no_change) {
	BOOST_CHECK(!plugin.proc);
	plugin.run(N);
	BOOST_CHECK(!plugin.proc);
	BOOST_TEST_CHECKPOINT("After first run");
	BOOST_CHECK(plugin.new_proc.valid());
	plugin.new_proc.wait();
	BOOST_TEST_CHECKPOINT("After ready");
	plugin.run(N);
	BOOST_CHECK(plugin.proc);
}

BOOST_AUTO_TEST_CASE(changed) {
	plugin.run(N);
	plugin.new_proc.wait();
	plugin.run(N);
	length = 2;
	BOOST_CHECK(!plugin.new_proc.valid());
	plugin.run(N);
	BOOST_CHECK(plugin.new_proc.valid());
	plugin.new_proc.wait();
	plugin.run(N);
	BOOST_CHECK(plugin.pars.length == 2);
}

BOOST_AUTO_TEST_CASE(buffer_changed) {
	plugin.run(N);
	plugin.new_proc.wait();
	plugin.run(N*2);
}

BOOST_AUTO_TEST_CASE(io_same) {
	plugin.connect_port(p_left_out, input[0]);
	plugin.connect_port(p_right_out, input[1]);
	plugin.run(N);
}

BOOST_AUTO_TEST_CASE(numbers) {
	auto rev = reverb(1000, 100, 2, 0.0001);
	BOOST_CHECK(rev.maxCoeff() < 5);
	BOOST_CHECK(rev.isFinite().all());
}

BOOST_AUTO_TEST_SUITE_END()
