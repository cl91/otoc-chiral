/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    chiral.cc

Abstract:

    This program computes the Green's function and the OTOC of the Chiral model.

Revision History:

    2023-10-10  File created

--*/

#include <ctime>
#include <memory>
#include "pcg_random.h"
#include <reapers.h>
#include "common.h"

using namespace REAPERS;
using namespace REAPERS::Model;
using namespace REAPERS::EvolutionAlgorithm;

template<RealScalar FpType>
using MatrixType = typename SumOps<FpType>::MatrixType;

// Argument parser for the CHIRAL simulation program.
struct ChiralArgParser : ArgParser {
    bool dump_ham;
    bool want_greenfcn;
    bool want_otoc;
    bool truncate_fp64;
    bool use_mt19937;
    bool seed_from_time;
    bool exact_diag;
    bool swap;
    float beta;
    float kry_tol;
    float param_u;
    float param_v;
    int idx0;
    int idx1;
    int N;

private:
    void parse_hook() {
	optdesc.add_options()
	    ("num_ops,N", progopts::value<int>(&N)->required(),
             "Specify the parameter N of the model.")
	    ("green", progopts::bool_switch(&want_greenfcn)->default_value(false),
	     "Compute the Green's function.")
	    ("otoc", progopts::bool_switch(&want_otoc)->default_value(false),
	     "Compute the OTOC. If neither --green or --otoc is specified, compute both.")
	    ("dump-ham", progopts::bool_switch(&dump_ham)->default_value(false),
	     "Dump the Hamiltonians constructed onto disk.")
	    ("beta,b", progopts::value<float>(&beta)->required(),
	     "Specifies the inverse temperature of the simulation.")
	    ("truncate-fp64", progopts::bool_switch(&truncate_fp64)->default_value(false),
	     "When both --fp32 and --fp64 are specified, truncate the fp64"
	     " couplings and random states to fp32, rather than promoting"
	     " fp32 to fp64. By default, fp32 couplings and random states"
	     " are promoted to fp64.")
	    ("use-mt19937", progopts::bool_switch(&use_mt19937)->default_value(false),
	     "Use the MT19937 from the C++ standard library as the pRNG."
	     " By default, PCG64 is used.")
	    ("index0", progopts::value<int>(&idx0)->required(),
	     "Specifies the first operator index in the OTOC.")
	    ("index1", progopts::value<int>(&idx1)->required(),
	     "Specifies the second operator index in the OTOC.")
	    ("param-u", progopts::value<float>(&param_u)->required(),
	     "Specifies the parameter u.")
	    ("param-v", progopts::value<float>(&param_v)->required(),
	     "Specifies the parameter v.")
	    ("tol", progopts::value<float>(&kry_tol)->default_value(0.0),
	     "Specifies the tolerance of the Krylov algorithm. The default is"
	     " machine precision.")
	    ("seed-from-time",
	     progopts::bool_switch(&seed_from_time)->default_value(false),
	     "Seed the pRNG using system time. By default, we use std::random_device"
	     " from the C++ stanfard library. Enable this if your C++ implementation's"
	     " std::random_device is broken (eg. deterministic).")
	    ("exact-diag", progopts::bool_switch(&exact_diag)->default_value(false),
	     "Use exact diagonalization rather than the Krylov method to compute"
	     " the state evolution exp(-iHt).")
	    ("swap", progopts::bool_switch(&swap)->default_value(false),
	     "Offload the initial state from VRAM to host memory.");
    }

    bool optcheck_hook() {
	// If user didn't specify Green fcn or OTOC, compute both
	if (!want_greenfcn && !want_otoc) {
	    want_greenfcn = want_otoc = true;
	}

	if (N <= 1) {
	    std::cerr << "N must be at least two." << std::endl;
	    return false;
	}

	if (truncate_fp64 && !(fp32 && fp64)) {
	    std::cerr << "You cannot specify --truncate-fp64 unless you specified"
		" both --fp32 and --fp64." << std::endl;
	    return false;
	}

	if (exact_diag && trace) {
	    std::cerr << "You cannot specify both --exact-diag and --trace."
		      << std::endl;
	    return false;
	}
	return true;
    }
};

template<RealScalar Fp>
class Chiral {
    int N;
    float param_u;
    float param_v;
    int idx0;
    int idx1;
public:
    using HamOp = SubspaceView<SumOps<Fp>>;
    using FpType = Fp;
    using StateType = State<FpType>;
    Chiral(const ChiralArgParser &args)
	: N(args.N), param_u(args.param_u), param_v(args.param_v),
	  idx0(args.idx0), idx1(args.idx1) {}
private:
    SumOps<FpType> sigma_x(int i) const {
	return SpinOp<Fp>::sigma_x(i);
    }
    SumOps<FpType> sigma_y(int i) const {
	return SpinOp<Fp>::sigma_y(i);
    }
    SumOps<FpType> sigma_z(int i) const {
	return SpinOp<Fp>::sigma_z(i);
    }
    SpinOp<Fp> op(int i) const {
	return SpinOp<Fp>::sigma_x(i);
    }
public:
    template<typename RandGen>
    HamOp gen_ham(RandGen &rg) const {
	HamOp ham;
	for (int i = 0; i < N-1; i++) {
	    auto ops = sigma_x(i) * sigma_x(i+1) + sigma_y(i) * sigma_y(i+1);
	    ham += (-param_u / 8) * HamOp(N, ops);
	}
	for (int i = 0; i < N-2; i++) {
	    auto ops = sigma_x(i) * (sigma_y(i+1) * sigma_z(i+2) -
				     sigma_z(i+1) * sigma_y(i+2)) -
		sigma_y(i) * (sigma_x(i+1) * sigma_z(i+2) - sigma_z(i+1) * sigma_x(i+2)) +
		sigma_z(i) * (sigma_x(i+1) * sigma_y(i+2) - sigma_y(i+1) * sigma_x(i+2));
	    ham += (param_v / 32) * HamOp(N, ops);
	}
	return ham;
    }
    int spin_chain_length() const { return N; }
    int dim() const { return 1ULL << N; }
    HamOp op_0() const { return HamOp(N, op(idx0)); }
    HamOp op_1() const { return HamOp(N, op(idx1)); }
};

template<RealScalar FpType>
using HamOp = typename Chiral<FpType>::HamOp;

// Base evaluator class for n-point functions which computes a single disorder
// realization.
template<RealScalar FpType>
class BaseEval {
    virtual void pre_evolve(const SumOps<FpType> &ham, State<FpType> &s,
			    FpType beta) = 0;
    virtual void evolve(const Chiral<FpType> &chiral, const HamOp<FpType> &ham,
			State<FpType> &s, FpType t, FpType beta) = 0;
    virtual complex<FpType> evolve_trace(const Chiral<FpType> &chiral, const HamOp<FpType> &ham,
					 const MatrixType<FpType> expmbH, FpType t) = 0;

protected:
    const ChiralArgParser &args;

    ~BaseEval() {}

    void evolve_step(const SumOps<FpType> &ham, State<FpType> &s,
		     FpType t, FpType beta) {
	if (args.exact_diag) {
	    s.template evolve<ExactDiagonalization>(ham, t, beta);
	} else if (args.kry_tol != 0.0) {
	    s.evolve(ham, t, beta, args.krylov_dim, (FpType)args.kry_tol);
	} else {
	    s.evolve(ham, t, beta, args.krylov_dim);
	}
    }

public:
    std::string name;

    BaseEval(const char *name, const ChiralArgParser &args) : args(args), name(name) {}

    // Evaluate the n-point function from 0 to t_max. In the case where trace
    // is false (the default), we compute the inner product of s0 and s, where
    // s0 is the result of pre_evolve() applied to a random state, and s is the
    // result of evolve() applied to s0. Note that in this case, before calling
    // this function you must set s0 to a random state, and the state s0 is
    // modified during the call so if you need the original initial state, you
    // need to make a copy of it before calling this function. This is to minimize
    // memory allocation overhead. If trace is set to true, this function
    // computes exp(-beta H/4) as a matrix and then calls evolve_trace(), which
    // the child classes will override to compute the final n-point function by
    // tracing over the n-point operator. In both cases results will be written
    // to vector v (after normalizing them by dividing by v[0]) as well as file
    // outf, and accumulated into sum.
    void eval(const Chiral<FpType> &chiral, const HamOp<FpType> &ham,
	      std::unique_ptr<State<FpType>> &s0,
	      std::vector<complex<FpType>> &v, std::ofstream &outf,
	      std::vector<complex<FpType>> &sum, Logger &logger) {
	auto start_time = time(nullptr);
	auto current_time = start_time;
	logger << "Running fp" << sizeof(FpType)*8 << " calculation." << endl;
	FpType dt = args.tmax / args.nsteps;
	if (args.trace) {
	    assert(!s0);
	    auto expmbH = ham.matexp(-args.beta);
	    for (int i = 0; i <= args.nsteps; i++) {
		v[i] = evolve_trace(chiral, ham, expmbH, i*dt);
		auto tm = time(nullptr);
		logger << "Time step " << i*dt << " done. Time for this step: "
		       << tm - current_time << "s"
		       << ". Total runtime so far for this disorder: "
		       << tm - start_time << "s." << endl;
		current_time = tm;
	    }
	} else {
	    assert(s0);
	    pre_evolve(ham, *s0, args.beta);
	    s0->gc();
	    std::unique_ptr<State<FpType,CPUImpl>> hostst;
	    if (args.swap) {
		hostst = std::make_unique<State<FpType,CPUImpl>>(*s0);
	    }
	    auto tm = time(nullptr);
	    logger << "Pre-evolve done. Time spent: "
		   << tm - current_time << "s." << endl;
	    current_time = tm;
	    for (int i = 0; i <= args.nsteps; i++) {
		State<FpType> s(*s0);
		if (args.swap) {
		    s0.reset();
		}
		evolve(chiral, ham, s, i*dt, args.beta);
		if (args.swap) {
		    s.gc();
		    s0 = std::make_unique<State<FpType>>(*hostst);
		}
		v[i] = (*s0) * s;
		auto tm = time(nullptr);
		logger << "Time step " << i*dt << " done. Time for this step: "
		       << tm - current_time << "s"
		       << ". Total runtime so far for this disorder: "
		       << tm - start_time << "s." << endl;
		current_time = tm;
	    }
	}
	auto v0 = v[0];
	for (int i = 0; i <= args.nsteps; i++) {
	    v[i] /= v0;
	    outf << dt*i << " " << v[i].real() << " " << v[i].imag() << std::endl;
	    sum[i] += v[i];
	}
    }
};

// This is the evaluator class for the Green's function.
template<RealScalar FpType>
class Green : public BaseEval<FpType> {
    void pre_evolve(const SumOps<FpType> &ham, State<FpType> &s, FpType beta) {
        this->evolve_step(ham, s, 0, beta/2);
    }

    void evolve(const Chiral<FpType> &chiral, const HamOp<FpType> &ham,
		State<FpType> &s, FpType t, FpType beta) {
	s *= chiral.op_0();
        this->evolve_step(ham, s, t, 0);
	s *= chiral.op_1();
	this->evolve_step(ham, s, -t, 0);
    }

    complex<FpType> evolve_trace(const Chiral<FpType> &chiral, const HamOp<FpType> &ham,
				 const MatrixType<FpType> expmbH, FpType t) {
	auto m0 = ham.matexp({0,-t});
	auto m1 = ham.matexp({0,t});
	auto op = chiral.op_0().get_matrix();
	return (m1 * op * m0 * op * expmbH).trace();
    }

public:
    Green(const ChiralArgParser &args) : BaseEval<FpType>("Green", args) {}
};

// This is the evaluator class for the OTOC.
template<RealScalar FpType>
class OTOC : public BaseEval<FpType> {
    void pre_evolve(const SumOps<FpType> &ham, State<FpType> &s, FpType beta) {
        this->evolve_step(ham, s, 0, beta/8);
    }

    void evolve(const Chiral<FpType> &chiral, const HamOp<FpType> &ham, State<FpType> &s,
		FpType t, FpType beta) {
	s *= chiral.op_1();
	this->evolve_step(ham, s, t, beta/4);
	s *= chiral.op_0();
	this->evolve_step(ham, s, -t, beta/4);
	s *= chiral.op_1();
	this->evolve_step(ham, s, t, beta/4);
	s *= chiral.op_0();
	this->evolve_step(ham, s, -t, 0);
    }

    complex<FpType> evolve_trace(const Chiral<FpType> &chiral, const HamOp<FpType> &ham,
				 const MatrixType<FpType> expmbH, FpType t) {
	auto m0 = ham.matexp({0,-t});
	auto m1 = ham.matexp({0,t});
	MatrixType<FpType> op0 = m1 * chiral.op_0().get_matrix() * m0;
	auto op1 = chiral.op_1().get_matrix();
	MatrixType<FpType> op01 = op0 * op1;
	return (op01 * op01 * expmbH).trace();
    }

public:
    OTOC(const ChiralArgParser &args) : BaseEval<FpType>("OTOC", args) {}
};

// Helper class which execute the computational tasks represented by
// the evaluator classes. We inherit from ChiralArgParser so we don't
// have to write args.??? in front of the command line parameters.
class Runner : protected ChiralArgParser {
    // Compute all disorder realization for the given n-point function
    template<template<typename> typename Eval, typename RandGen>
    void runjob(RandGen &rg) {
	std::stringstream ss;
	ss << "N" << N << "M" << M << "beta" << beta
	   << "u" << param_u << "v" << param_v
	   << "tmax" << tmax << "nsteps" << nsteps
	   << "idx" << idx0 << "idx" << idx1;
	if (!exact_diag && !trace) {
	    ss << "krydim" << krylov_dim;
	    if (kry_tol != 0.0) {
		ss << "krytol" << kry_tol;
	    }
	}
	if (exact_diag) {
	    ss << "ed";
	}
	if (trace) {
	    ss << "trace";
	}
	Eval<float> eval32(*this);
	Eval<double> eval64(*this);
	auto jobname = eval32.name;
	auto outfname = jobname + ss.str();
	auto avgfname = jobname + "Avg" + ss.str();
	std::ofstream outf32, outf64, outfdiff;
	std::ofstream avgf32, avgf64, avgfdiff;
	if (fp32 && fp64) {
	    if (truncate_fp64) {
		outfname += "crstrunc";
		avgfname += "crstrunc";
	    } else {
		outfname += "crsprom";
		avgfname += "crsprom";
	    }
	}
	if (fp32) {
	    outf32.rdbuf()->pubsetbuf(0, 0);
	    outf32.open(outfname + "fp32");
	    avgf32.rdbuf()->pubsetbuf(0, 0);
	    avgf32.open(avgfname + "fp32");
	}
	if (fp64) {
	    outf64.rdbuf()->pubsetbuf(0, 0);
	    outf64.open(outfname + "fp64");
	    avgf64.rdbuf()->pubsetbuf(0, 0);
	    avgf64.open(avgfname + "fp64");
	}
	if (fp32 && fp64) {
	    outfdiff.rdbuf()->pubsetbuf(0, 0);
	    outfdiff.open(outfname + "diff");
	    avgfdiff.rdbuf()->pubsetbuf(0, 0);
	    avgfdiff.open(avgfname + "diff");
	}
	std::ofstream hamf;
	if (dump_ham) {
	    hamf.rdbuf()->pubsetbuf(0, 0);
	    hamf.open(std::string("Ham") + outfname);
	}
	std::vector<complex<float>> v32(nsteps+1);
	std::vector<complex<double>> v64(nsteps+1);
	std::vector<double> vdiff(nsteps+1);
	std::vector<complex<float>> sum32(nsteps+1);
	std::vector<complex<double>> sum64(nsteps+1);
	std::vector<double> sumdiff(nsteps+1);
	std::ofstream logf;
	logf.rdbuf()->pubsetbuf(0, 0);
	logf.open(std::string("Log") + jobname + ss.str());
	Logger logger(verbose, logf);
	logger << "Running " << jobname << " calculation using build "
	       << GITHASH << ".\nParameters are: " << ss.str() << endl;
	std::unique_ptr<State<float>> init32;
	std::unique_ptr<State<double>> init64;
	Chiral<float> chiral32(*this);
	Chiral<double> chiral64(*this);
	if (fp32 && !trace) {
	    init32 = std::make_unique<State<float>>(chiral32.spin_chain_length());
	}
	if (fp64 && !trace) {
	    init64 = std::make_unique<State<double>>(chiral64.spin_chain_length());
	}
	HamOp<float> ham32;
	HamOp<double> ham64;
	double dt = tmax / nsteps;
	for (int u = 0; u < M; u++) {
	    logger << "Computing disorder " << u << endl;
	    if (!fp32 || truncate_fp64) {
		if (!trace) {
		    init64->random_state();
		}
		ham64 = chiral64.gen_ham(rg);
	    } else {
		if (!trace) {
		    init32->random_state();
		}
		ham32 = chiral32.gen_ham(rg);
	    }
	    if (dump_ham) {
		if (!fp32 || truncate_fp64) {
		    hamf << "H = " << ham64 << std::endl;
		} else {
		    hamf << "H = " << ham32 << std::endl;
		}
	    }
	    // Compute one single disorder realization.
	    if (fp32) {
		// If fp64 is also requested, we should make sure we evolve the
		// states using the same Hamiltonian and same initial state as
		// fp64. There are two options here. We can either promote
		// the fp32 Hamiltonian and initial states to fp64, or
		// truncating fp64 into fp32. If user specified to truncate,
		// do the truncation now.
		if (fp64 && truncate_fp64) {
		    if (!trace) {
			*init32 = *init64;
		    }
		    ham32 = ham64;
		}
		eval32.eval(chiral32, ham32, init32, v32, outf32, sum32, logger);
	    }
	    if (fp64) {
		// If user specfied both --fp32 and --fp64 but did not specify
		// --truncate-fp64, promote the fp32 Hamiltonian and initial states
		// to fp64 instead.
		if (fp32 && !truncate_fp64) {
		    if (!trace) {
			*init64 = *init32;
		    }
		    ham64 = ham32;
		}
		eval64.eval(chiral64, ham64, init64, v64, outf64, sum64, logger);
	    }
	    if (fp32 && fp64) {
		// If user requested both fp32 and fp64, also compute the difference
		for (int i = 0; i <= nsteps; i++) {
		    vdiff[i] = abs(v64[i] - complex<double>(v32[i]));
		    outfdiff << dt*i << " " << vdiff[i] << std::endl;
		    sumdiff[i] += vdiff[i];
		}
	    }
	}
	for (int i = 0; i <= nsteps; i++) {
	    if (fp32) {
		sum32[i] /= M;
		avgf32 << dt*i << " " << sum32[i].real() << " "
		       << sum32[i].imag() << std::endl;
	    }
	    if (fp64) {
		sum64[i] /= M;
		avgf64 << dt*i << " " << sum64[i].real() << " "
		       << sum64[i].imag() << std::endl;
	    }
	    if (fp32 && fp64) {
		sumdiff[i] /= M;
		avgfdiff << dt*i << " " << sumdiff[i] << std::endl;
	    }
	}
    }

public:
    int run(int argc, const char *argv[]) {
	if (!parse(argc, argv)) {
	    return 1;
	}
	unsigned int seed = 0;
	if (seed_from_time) {
	    seed = time(nullptr);
	} else {
	    std::random_device rd;
	    seed = rd();
	}
	if (use_mt19937) {
	    // The user requested mt19937 as the pRNG. Use it.
	    std::mt19937 rg(seed);
	    if (want_otoc) {
		runjob<OTOC>(rg);
	    }
	    if (want_greenfcn) {
		runjob<Green>(rg);
	    }
	} else {
	    // Use PCG64 as the random number engine
	    pcg64 rg(seed);
	    if (want_otoc) {
		runjob<OTOC>(rg);
	    }
	    if (want_greenfcn) {
		runjob<Green>(rg);
	    }
	}
	return 0;
    }
};

int main(int argc, const char *argv[])
{
    try {
	std::cout << "Human. This is CHIRAL"
#ifndef REAPERS_NOGPU
		  << "-GPU"
#endif
#ifdef __INTEL_LLVM_COMPILER
		  << "-AVX"
#endif
		  << "-" GITHASH ", powered by the REAPERS library.\n" << std::endl;
	Runner runner;
	runner.run(argc, argv);
    } catch (const std::exception &e) {
	std::cerr << "Program terminated abnormally due to the following error:"
		  << std::endl << e.what() << std::endl;
	return 1;
    } catch (...) {
	std::cerr << "Program terminated abnormally due an unknown exception."
		  << std::endl;
	return 1;
    }
    return 0;
}
