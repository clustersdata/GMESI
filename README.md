# GMESI
GMESI: A New Software for Searching the Global-Minimum Spin Multiplicity of Electron-Spin State Isomers of Clusters

# GMESI (v1.0) Python Implementation

This python code implements core algorithmic workflow of **GMESI** software: including initial spin‑multiplicity generation, random spin perturbation, duplicate spin recognition, four global‑optimization engines: BH (Basin‑Hopping), IBH (Improved Basin‑Hopping), GA (Genetic Algorithm), PSO (Particle Swarm Optimization).

> 
> Important note:
> 
> 
> 1. This is algorithm‑level skeleton implementation. Real energy evaluation calls external quantum‑chemistry backend (xTB / ORCA / PySCF etc). No built‑in electronic‑structure solver.
> 2. Spin multiplicity $((M = 2S+1))$, must be positive integer.
> 3. Metropolis criterion for BH/IBH, adaptive step size for IBH, discrete PSO, integer‑crossover GA are reproduced strictly follow formulas in paper.
> 4. Convergence rule: lowest spin state unchanged for `conv_cycles` successive iterations.

```
"""
GMESI: A New Software for Searching the Global‑Minimum Spin Multiplicity
of Electron‑Spin State Isomers of Clusters

Reference Paper: GMESI: A New Software for Searching the Global‑Minimum Spin Multiplicity
of Electron‑Spin State Isomers of Clusters

License: Academic Non‑Commercial Use
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Callable

random.seed(42)

class GMESI:
    def __init__(
        self,
        min_spin_mult: int,
        max_spin_mult: int,
        energy_backend: Callable[[int], float],
        algorithm: str = "IBH",
        init_step_delta: int = 2,
        temperature: float = 10.0,
        target_accept_rate: float = 0.4,
        alpha: float = 0.15,
        beta: float = 0.08,
        conv_cycles: int = 4,
        max_iter: int = 50,
        pop_size: int = 12,
        w_pso: float = 0.7,
        c1_pso: float = 1.4,
        c2_pso: float =1.4,
        mutation_rate: float=0.2
    ):
        """
        Initialize GMESI spin‑multiplicity global‑search instance

        Parameters
        ----------
        min_spin_mult : int
            Minimum physically allowed spin multiplicity \(\(M = 2S+1\)\) (>=1, singlet M=1)
        max_spin_mult : int
            Maximum physically feasible spin multiplicity for cluster system
        energy_backend : Callable[[int], float]
            External quantum‑chemistry backend function: input spin multiplicity M(int),
            returns total electronic energy (kcal/mol). Wrapper for spGFN2‑xTB / ORCA / PySCF.
        algorithm : str
            Choose global optimizer: ["BH", "IBH", "PSO", "GA"]
                BH: Standard Basin‑Hopping
                IBH: Improved Basin‑Hopping (Recommended Default in paper)
                PSO: Particle Swarm Optimization
                GA: Genetic Algorithm
        init_step_delta : int
            Initial perturbation step δ for spin multiplicity random perturbation
        temperature : float
            Initial effective temperature T for Metropolis acceptance (BH/IBH)
        target_accept_rate : float
            Target acceptance rate for adaptive step‑size (IBH only, 0~1)
        alpha : float
            Learning rate for adaptive δ update (IBH formula 5)
        beta : float
            Temperature decay coefficient T_iter = T0 * exp(-beta*iter) (IBH)
        conv_cycles : int
            Convergence criterion: if best energy unchanged for conv_cycles iterations → converged
        max_iter : int
            Upper limit of total iterations, hard stop to avoid infinite loop
        pop_size : int
            Population size for PSO / GA
        w_pso : float
            Inertia weight ω for discrete PSO
        c1_pso : float
            Cognitive acceleration coefficient c1 for PSO
        c2_pso : float
            Social acceleration coefficient c2 for PSO
        mutation_rate : float
            Mutation probability for GA algorithm
        """
        self.M_min: int = min_spin_mult
        self.M_max: int = max_spin_mult
        self.energy_backend = energy_backend
        assert self.M_min >= 1
        assert self.M_max > self.M_min
        assert algorithm in ["BH", "IBH", "PSO", "GA"]
        self.algorithm = algorithm

        # spin perturbation hyper‑params
        self.delta: int = init_step_delta
        self.T0: float = temperature
        self.T: float = temperature
        self.target_acc_rate = target_accept_rate
        self.alpha = alpha
        self.beta = beta

        self.conv_cycles = conv_cycles
        self.max_iter = max_iter

        # PSO parameters
        self.pop_size = pop_size
        self.w_pso = w_pso
        self.c1_pso = c1_pso
        self.c2_pso = c2_pso

        # GA parameters
        self.mutation_rate = mutation_rate

        # Runtime storage: duplicate recognition set M_computed
        self.M_computed: set[int] = set()
        self.energy_db: Dict[int, float] = dict()

        # global best record
        self.best_M: Optional[int] = None
        self.best_energy: Optional[float] = None
        self.best_streak: int = 0

        # iteration log
        self.iter_log: List[dict] = []
        self.accept_history: List[bool] = []

    def _constrain_M(self, m_candidate: int) -> int:
        """Force spin multiplicity to valid integer range [M_min, M_max]"""
        m = int(round(m_candidate))
        return max(self.M_min, min(self.M_max, m))

    def generate_initial_spin_multiplicity(self) -> List[int]:
        """
        Generate chemically reasonable initial spin multiplicity set.
        based on valence‑electron, charge, metal oxidation state.
        Sample several points across M_min ~ M_max
        """
        init_list = []
        num_sample = max(4, int((self.M_max - self.M_min)/2)+1)
        for _ in range(num_sample):
            m = random.randint(self.M_min, self.M_max)
            init_list.append(m)
        # ensure boundary values are included
        init_list.append(self.M_min)
        init_list.append(self.M_max)
        return sorted(list(set(init_list)))

    def spin_random_perturb(self, M_parent: int, k: int =1) -> int:
        """
        Random spin multiplicity perturbation (Formula 1 & 2 from paper):
        M_child = M_parent + k * δ * ξ
        ξ ∈ {-1, +1}
        """
        xi = random.choice([-1, 1])
        m_child = M_parent + k * self.delta * xi
        return self._constrain_M(m_child)

    def get_energy(self, M: int) -> float:
        """
        Duplicate recognition wrapper: check if spin multiplicity M already computed.
        If exists: return cached energy; else call backend, store into database.
        """
        if M in self.M_computed:
            return self.energy_db[M]
        e = self.energy_backend(M)
        self.M_computed.add(M)
        self.energy_db[M] = e
        return e

    def metropolis_accept(self, e_new: float, e_cur: float) -> Tuple[bool, float]:
        """
        Metropolis acceptance probability (Formula4):
        P_accept = 1 if E_new < E_cur
        P_accept = exp(-(E_new‑E_cur)/T) if E_new >= E_cur
        """
        if e_new < e_cur:
            return True, 1.0
        delta_e = e_new - e_cur
        prob = math.exp(- delta_e / self.T)
        r = random.random()
        return r < prob, prob

    def _update_best(self, M_candidate: int, e_candidate: float):
        """Update global best spin multiplicity & convergence streak counter"""
        if self.best_M is None:
            self.best_M = M_candidate
            self.best_energy = e_candidate
            self.best_streak = 1
        else:
            if e_candidate < self.best_energy:
                self.best_M = M_candidate
                self.best_energy = e_candidate
                self.best_streak = 1
            else:
                self.best_streak += 1

    def run_bh(self) -> Tuple[int, float, List[dict]]:
        """Standard Basin‑Hopping algorithm (BH)"""
        M_current = random.choice(self.generate_initial_spin_multiplicity())
        e_current = self.get_energy(M_current)
        self._update_best(M_current, e_current)

        for it in range(self.max_iter):
            M_trial = self.spin_random_perturb(M_current)
            e_trial = self.get_energy(M_trial)
            accept, _ = self.metropolis_accept(e_trial, e_current)
            self.accept_history.append(accept)
            if accept:
                M_current = M_trial
                e_current = e_trial

            self._update_best(M_current, e_current)
            self.iter_log.append({
                "iter": it,
                "current_M": M_current,
                "current_E": e_current,
                "global_best_M": self.best_M,
                "global_best_E": self.best_energy,
                "T": self.T
            })
            if self.best_streak >= self.conv_cycles:
                break
        return self.best_M, self.best_energy, self.iter_log

    def run_ibh(self) -> Tuple[int, float, List[dict]]:
        """
        Improved Basin‑Hopping (IBH):
            1. Adaptive step‑size δ update (Formula 5)
            2. Exponential dynamic temperature decay T_iter = T0*exp(-β·iter)
        """
        M_current = random.choice(self.generate_initial_spin_multiplicity())
        e_current = self.get_energy(M_current)
        self._update_best(M_current, e_current)
        accept_window = 8

        for it in range(self.max_iter):
            self.T = self.T0 * math.exp(- self.beta * it)
            M_trial = self.spin_random_perturb(M_current)
            e_trial = self.get_energy(M_trial)
            accept, _ = self.metropolis_accept(e_trial, e_current)
            self.accept_history.append(accept)

            # Adaptive delta (Formula 5)
            if len(self.accept_history) >= accept_window:
                recent_acc = sum(self.accept_history[-accept_window:]) / accept_window
                self.delta = int(round(self.delta * (1 + self.alpha * (recent_acc - self.target_acc_rate))))
                self.delta = max(1, self.delta)

            if accept:
                M_current = M_trial
                e_current = e_trial

            self._update_best(M_current, e_current)
            self.iter_log.append({
                "iter": it,
                "current_M": M_current,
                "current_E": e_current,
                "global_best_M": self.best_M,
                "global_best_E": self.best_energy,
                "T": self.T,
                "delta": self.delta
            })
            if self.best_streak >= self.conv_cycles:
                break
        return self.best_M, self.best_energy, self.iter_log

    def run_pso(self) -> Tuple[int, float, List[dict]]:
        """
        Discrete‑adapted Particle Swarm Optimization (PSO), Formula 6 &7
        Each particle stores spin multiplicity M_i
        velocity update → round to integer spin multiplicity
        """
        init_pool = self.generate_initial_spin_multiplicity()
        particles = []
        personal_best_M = []
        personal_best_E = []
        velocities = []

        for _ in range(self.pop_size):
            m0 = random.choice(init_pool)
            e0 = self.get_energy(m0)
            particles.append(m0)
            personal_best_M.append(m0)
            personal_best_E.append(e0)
            velocities.append(0.0)

        global_best_idx = int(np.argmin(personal_best_E))
        global_best_M = personal_best_M[global_best_idx]
        global_best_E = personal_best_E[global_best_idx]
        self.best_M = global_best_M
        self.best_energy = global_best_E

        for it in range(self.max_iter):
            for i in range(self.pop_size):
                r1 = random.random()
                r2 = random.random()
                v_new = (self.w_pso * velocities[i]
                         + self.c1_pso * r1 * (personal_best_M[i] - particles[i])
                         + self.c2_pso * r2 * (global_best_M - particles[i]))
                velocities[i] = v_new
                m_new = round(particles[i] + v_new)
                m_new = self._constrain_M(m_new)
                e_new = self.get_energy(m_new)
                particles[i] = m_new
                if e_new < personal_best_E[i]:
                    personal_best_M[i] = m_new
                    personal_best_E[i] = e_new
                if e_new < global_best_E:
                    global_best_M = m_new
                    global_best_E = e_new
                    self.best_streak =0
            self.best_M = global_best_M
            self.best_energy = global_best_E
            self.iter_log.append({
                "iter": it,
                "global_best_M": self.best_M,
                "global_best_E": self.best_energy
            })
            self.best_streak += 1
            if self.best_streak >= self.conv_cycles:
                break
        return self.best_M, self.best_energy, self.iter_log

    def run_ga(self) -> Tuple[int, float, List[dict]]:
        """
        Genetic Algorithm (GA): selection, integer crossover (Formula 8), mutation.
        Fitness = -energy (minimization → maximization problem)
        M_child = floor(γ * M_parent1 + (1‑γ)*M_parent2 ), γ ∈ [0,1]
        """
        init_pool = self.generate_initial_spin_multiplicity()
        population = [random.choice(init_pool) for _ in range(self.pop_size)]
        # evaluate initial pop
        for m in population:
            _ = self.get_energy(m)

        def fitness(m):
            return -1.0 * self.energy_db[m]

        for it in range(self.max_iter):
            fit_list = [fitness(m) for m in population]
            total_fit = sum(fit_list)
            # selection: roulette‑wheel
            new_pop = []
            for _ in range(self.pop_size):
                r = random.random() * total_fit
                acc = 0
                sel_idx =0
                for idx, fv in enumerate(fit_list):
                    acc += fv
                    if acc >= r:
                        sel_idx = idx
                        break
                new_pop.append(population[sel_idx])
            # crossover
            next_pop = []
            for i in range(0, self.pop_size,2):
                p1 = new_pop[i]
                p2 = new_pop[i+1 if i+1<self.pop_size else 0]
                gamma = random.random()
                m_child1 = math.floor(gamma * p1 + (1.0 - gamma)*p2)
                m_child2 = math.floor(gamma * p2 + (1.0 - gamma)*p1)
                m_child1 = self._constrain_M(m_child1)
                m_child2 = self._constrain_M(m_child2)
                next_pop.append(m_child1)
                next_pop.append(m_child2)
            # mutation
            for idx in range(len(next_pop)):
                if random.random() < self.mutation_rate:
                    next_pop[idx] = self.spin_random_perturb(next_pop[idx])
            population = next_pop
            # evaluate new population
            for m in population:
                _ = self.get_energy(m)
            # update global best
            current_best_m = min(population, key=lambda x:self.energy_db[x])
            current_best_e = self.energy_db[current_best_m]
            self._update_best(current_best_m, current_best_e)
            self.iter_log.append({
                "iter": it,
                "global_best_M": self.best_M,
                "global_best_E": self.best_energy
            })
            if self.best_streak >= self.conv_cycles:
                break
        return self.best_M, self.best_energy, self.iter_log

    def execute_search(self):
        """Dispatch to selected optimization algorithm"""
        if self.algorithm == "BH":
            return self.run_bh()
        elif self.algorithm == "IBH":
            return self.run_ibh()
        elif self.algorithm == "PSO":
            return self.run_pso()
        elif self.algorithm == "GA":
            return self.run_ga()

# numpy dependency for PSO
try:
    import numpy as np
except ImportError:
    raise ImportError("Please install numpy: pip install numpy, required for PSO algorithm of GMESI")

if __name__ == "__main__":
    # ------------------------------
    # Example: Mock backend for Pt6+ cluster test case (Paper Table 2 spGFN2‑xTB relative energies)
    # Real usage: replace mock_backend with wrapper calling spGFN2‑xTB / ORCA / PySCF
    # ------------------------------
    def mock_pt6plus_backend(M: int) -> float:
        """Mock energy backend for Pt6+ cluster (GM M=6, energy=0 kcal/mol reference)"""
        energy_map = {
            2:4.71,
            4:3.10,
            6:0.00,
            8:0.21,
            10:46.26,
            12:92.40,
            14:176.40
        }
        if M not in energy_map:
            return 200.0
        return energy_map[M]

    print("==== GMESI Demo: Pt6+ Cluster (Ground‑state spin multiplicity M=6) ====")
    gmesi_run = GMESI(
        min_spin_mult=2,
        max_spin_mult=14,
        energy_backend=mock_pt6plus_backend,
        algorithm="IBH",
        conv_cycles=3,
        max_iter=30
    )
    best_M, best_E, log = gmesi_run.execute_search()
    print(f"Global‑Minimum Spin Multiplicity (2S+1): {best_M}")
    print(f"Relative Energy (kcal/mol): {best_E:.2f}")
    print(f"Total unique spin multiplicities evaluated: {len(gmesi_run.M_computed)}")
    print("Iteration log snippet:")
    for entry in log[:6]:
        print(entry)
```

## Detailed Usage Document for GMESI Python Package

### 1. Overview

GMESI automates global‑minimum spin‑multiplicity search for electron‑spin state isomers(ESSI).

> 
> **Core variables**: $M = 2S+1$, total spin multiplicity, positive integer.
> GMESI itself has no electronic‑structure solver. All energy evaluation delegated to external quantum‑chemistry backend function.

Supported global‑optimization algorithms:

| Algorithm | Full Name | Paper conclusion |
| --- | --- | --- |
| `IBH` | Improved Basin‑Hopping | **Recommended default**: fastest convergence, fewest iterations |
| `BH` | Standard Basin‑Hopping | Baseline method |
| `PSO` | Particle Swarm Optimization | Population‑based, slower for rugged spin PES |
| `GA` | Genetic Algorithm | Strong global search, highest computational overhead |

### 2. Environment Dependencies

```
pip install numpy
```

- numpy: required for PSO algorithm
- External quantum‑chemistry software: **xTB, ORCA6, PySCF, Molpro**. You need write a thin python wrapper function as `energy_backend`.

### 3. Core Class: `GMESI`

#### Constructor arguments

```
GMESI(
    min_spin_mult: int,          # lower physical bound of spin multiplicity (>=1, singlet=1)
    max_spin_mult: int,          # upper physical bound of spin multiplicity for your cluster
    energy_backend: Callable,     # critical backend function input M(int), return energy float (kcal/mol)
    algorithm="IBH",              # choose optimizer: "BH","IBH","PSO","GA"
    init_step_delta=2,            # initial spin perturbation step δ
    temperature=10.0,             # initial effective temperature T0 for Metropolis criterion
    target_accept_rate=0.4,       # IBH adaptive step target accept rate
    alpha=0.15,                   # IBH learning rate for δ adaption
    beta=0.08,                    # IBH temperature decay coefficient
    conv_cycles=4,                # convergence condition: best energy unchanged N iterations → stop
    max_iter=50,                  # hard upper iteration limit
    pop_size=12,                  # population size for PSO / GA
    w_pso=0.7, c1_pso=1.4,c2_pso=1.4, # PSO hyper‑parameters
    mutation_rate=0.2             # GA mutation probability
)
```

#### Member Attributes after running search

| Attribute | Description |
| --- | --- |
| `.best_M` | Global‑minimum spin multiplicity (2S+1) |
| `.best_energy` | Energy of global‑minimum spin state, returned by backend |
| `.M_computed` | `set`: all unique spin multiplicities evaluated (duplicate elimination result) |
| `.energy_db` | `dict[M,energy]` cache table for spin‑state energies |
| `.iter_log` | List of dictionary: full iteration history log for post‑analysis / plotting convergence curve |

#### Main Method

```
best_M, best_E, iteration_log = gmesi.execute_search()
```

Return tuple:

1. `best_M`: integer global‑minimum spin multiplicity
2. `best_E`: float energy from backend
3. `iteration_log`: runtime log for plotting convergence curve.

### 4. How to write your own `energy_backend` wrapper

The backend function is the bridge between GMESI and quantum‑chemistry software. Input spin multiplicity integer M, return total energy (kcal/mol).

#### Example 1: Wrapper for spGFN2‑xTB

```
import subprocess

def backend_sptb(M:int) -> float:
    """GMESI backend wrapper for spGFN2‑xTB
    Input spin multiplicity \(\(M = 2S+1\)\), return relative energy in kcal/mol
    """
    # write xTB input with spin‑polarized flag, set spin multiplicity M
    result = subprocess.run(cmd, capture_output=True, text=True)
    # parse output file "xtb.out" extract total energy (Hartree), convert Hartree → kcal/mol
    hartree_2_kcal = 627.509
    # parse code omitted for brevity
    total_energy_hartree = parse_xtb_output("xtb.out")
    return total_energy_hartree * hartree_2_kcal
```

#### Example2: Wrapper for ORCA PBE0‑D4/def2‑TZVP (high‑level refinement backend)

GMESI can drive ORCA for final refinement. You generate ORCA input template with spin multiplicity placeholder, GMESI fills M and submit jobs.

> 
> Note for strongly correlated systems (Fe4S4 clusters etc): you can replace ORCA template to CASSCF/CASPT2/NEVPT2 input; GMESI will enumerate spin‑multiplicity candidates, backend handles multi‑reference calculation.

### 5. Complete Real‑world Workflow Template

```
from gmesi_module import GMESI

# Step 1: define your quantum‑chemistry backend
def my_cluster_backend(M:int):
    # your wrapper for xTB / ORCA / PySCF
    ...

# Step 2: Initialize GMESI instance
runner = GMESI(
    min_spin_mult=1,
    max_spin_mult=21,
    energy_backend=my_cluster_backend,
    algorithm="IBH",
    conv_cycles=4,
    max_iter=40
)

# Step3: run global spin‑multiplicity search
gmsm_M, gmsm_E, log_data = runner.execute_search()

# Step4: print main result
print(f"Global‑Minimum Spin Multiplicity GMSM = {gmsm_M}")
print(f"Energy of GMSM spin state = {gmsm_E:.3f} kcal/mol")
print(f"Number of unique spin states computed: {len(runner.M_computed)}")

# Step5: save log for plotting convergence curve
import json
with open("gmesi_convergence_log.json","w") as f:
    json.dump(log_data, f, indent=2)
```

### 6. Important Limitations (Consistent with original paper)

1. Current implementation only optimizes **total spin multiplicity ((M = 2S+1))**. It does **NOT distinguish different electronic isomers sharing identical total multiplicity** (closed‑shell singlet vs broken‑symmetry open‑shell singlet; different local spin‑coupling pattern with same M). Future version will add composite duplicate‑key including `<S^2>`, Mulliken spin populations as described in "Limitations and future work".
2. GMESI does **not implement electronic‑structure calculation itself**. Prediction quality is fully bounded by accuracy of your chosen backend.
   - Fast screening: `spGFN2‑xTB` (recommended)
   - Refinement: DFT `PBE0‑D4/def2‑TZVP`
   - Strong static‑correlation systems: use multireference backend CASSCF/CASPT2/NEVPT2
3. No periodic boundary condition support, for molecular clusters / complexes only.
4. No spin‑orbit‑coupling Hamiltonian inside GMESI. Spin‑orbit effects must be enabled inside backend software template keywords. GMESI ranks energies returned by backend.

### 7. Algorithm Tuning Tips according to paper benchmark

1. Default choice for most cluster systems: `algorithm="IBH"`. As shown in Table1, IBH has minimal wall‑time and iteration counts for Pt6+ benchmark.
2. For high‑spin large clusters: increase `init_step_delta` for larger spin jumps; increase `max_iter`.
3. Convergence control: increase `conv_cycles` to 5‑6 for systems with near‑degenerate spin states (e.g. Pt6+, where M=6 and M=8 energy difference only ~0.2 kcal/mol under spGFN2‑xTB).
4. For GA/PSO: increase `pop_size` for complex heterometallic clusters, computational cost scales linearly with population size.

### 8. Benchmark Test

Run built‑in `__main__` demo code. It simulates Pt6+ cluster benchmark, expected output global‑minimum spin multiplicity is **6**. This verifies the whole algorithm pipeline works.

```
python GMESI.py
```

If you want to compare among BH/IBH/GA/PSO, you loop over four algorithms with same mock backend and record iteration count & wall‑clock time.

### 9. Extend to Future‑work features

We will extend this code to add new features:

1. Extend duplicate‑recognition key: combine M, `<S^2>` expectation value, Mulliken spin population to distinguish same‑multiplicity different electronic states.
2. Add Heisenberg exchange coupling constant fitting module from converged spin densities.
3. Add periodic backend interface for solid‑state spin global search.
4. Add finite‑temperature thermodynamic sampling module for spin‑crossover systems.
5. Interface machine‑learning spin‑energy potentials to replace quantum‑chemistry backend for ultra‑large nanoclusters.

> 
> Citation: If you use this algorithm implementation, please cite original GMESI paper.
