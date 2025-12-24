#!/usr/bin/env python3
"""
DREAMPlace Evaluator for LLM-based Hyperparameter Optimization

This evaluator executes DREAMPlace placer with given parameters and parses
the placement logs into structured JSON format for LLM analysis.

Based on actual DREAMPlace log format analysis.
"""

import json
import os
import re
import sys
import subprocess
import time
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class IterationMetrics:
    """Metrics for a single optimization iteration"""
    iteration: int
    detailed_step: Optional[Tuple[int, int, int]] = None
    objective: Optional[float] = None
    wirelength: Optional[float] = None
    density_weight: Optional[float] = None
    hpwl: Optional[float] = None
    overflow: Optional[float] = None
    max_density: Optional[float] = None
    gamma: Optional[float] = None
    eval_time: Optional[float] = None


@dataclass
class StagePerformance:
    """Performance metrics for each placement stage"""
    stage_name: str
    duration: float


@dataclass
class ConvergenceAnalysis:
    """Analysis of convergence patterns"""
    converged: bool
    convergence_rate: str  # "fast", "medium", "slow"
    oscillation_detected: bool
    early_convergence: bool
    final_iterations_stable: bool
    overflow_reduction_rate: float
    hpwl_reduction_rate: float
    total_iterations: int
    overflow_converged: bool  # overflow < 0.1
    density_stabilized: bool


@dataclass
class PlacementResult:
    """Complete placement result with structured metrics"""
    # Basic results
    final_hpwl: float
    final_iteration: int
    total_time: float
    success: bool
    error_message: Optional[str] = None

  
    # Optimization trace
    iterations: List[IterationMetrics] = None
    stage_performance: List[StagePerformance] = None
    convergence_analysis: ConvergenceAnalysis = None

  

class DREAMPlaceLogParser:
    """Parser for DREAMPlace placement logs based on actual log format"""

    def __init__(self):
        # Full iteration pattern (global placement)
        self.full_iteration_pattern = re.compile(
            r'iteration\s+(\d+),\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\),\s*Obj\s+([\d.+-E]+),\s*'
            r'DensityWeight\s+([\d.+-E]+),\s*wHPWL\s+([\d.+-E]+),\s*'
            r'Overflow\s+([\d.+-E]+),\s*MaxDensity\s+([\d.+-E]+),\s*gamma\s+([\d.+-E]+),\s*time\s+([\d.+-E]+)ms'
        )

        # Simple iteration pattern (legalization/detailed placement)
        self.simple_iteration_pattern = re.compile(
            r'iteration\s+(\d+),\s*wHPWL\s+([\d.+-E]+),\s*time\s+([\d.+-E]+)ms'
        )

        # Stage timing patterns
        self.timing_patterns = [
            (re.compile(r'reading benchmark takes ([\d.]+) seconds'), "reading_benchmark"),
            (re.compile(r'non-linear placement initialization takes ([\d.]+) seconds'), "initialization"),
            (re.compile(r'nesterov initialization takes ([\d.]+) seconds'), "optimizer_init"),
            (re.compile(r'optimizer nesterov takes ([\d.]+) seconds'), "global_placement"),
            (re.compile(r'legalization takes ([\d.]+) seconds'), "legalization"),
            (re.compile(r'detailed placement takes ([\d.]+) seconds'), "detailed_placement"),
            (re.compile(r'placement takes ([\d.]+) seconds'), "total_placement"),
            (re.compile(r'write SolutionFileFormat\.BOOKSHELF takes ([\d.]+) seconds'), "writing_results"),
        ]

        # Final results pattern
        self.final_results_pattern = re.compile(
            r'Final iteration: (\d+)\s+.*?Final HPWL: ([\d.+-E]+)'
        )

        # Stage start patterns
        self.stage_start_patterns = [
            (re.compile(r'Start legalization'), "legalization"),
            (re.compile(r'Start ABCDPlace for refinement'), "detailed_placement"),
            (re.compile(r'Global placement:'), "global_placement"),
        ]

    def parse_log(self, log_text: str) -> PlacementResult:
        """Parse DREAMPlace log into structured result"""
        lines = log_text.split('\n')

        iterations = []
        stage_performance = []
        final_iteration = None
        final_hpwl = None
        success = False
        error_message = None

        # Parse iterations
        current_stage_hpwl = None
        for i, line in enumerate(lines):
            # Parse full iteration (global placement)
            match = self.full_iteration_pattern.search(line)
            if match:
                iteration_data = IterationMetrics(
                    iteration=int(match.group(1)),
                    detailed_step=(int(match.group(2)), int(match.group(3)), int(match.group(4))),
                    objective=float(match.group(5)),
                    density_weight=float(match.group(6)),
                    hpwl=float(match.group(7)),
                    overflow=float(match.group(8)),
                    max_density=float(match.group(9)),
                    gamma=float(match.group(10)),
                    eval_time=float(match.group(11))
                )
                iterations.append(iteration_data)
                current_stage_hpwl = iteration_data.hpwl
                continue

            # Parse simple iteration (legalization/detailed placement)
            match = self.simple_iteration_pattern.search(line)
            if match:
                iteration_data = IterationMetrics(
                    iteration=int(match.group(1)),
                    hpwl=float(match.group(2)),
                    eval_time=float(match.group(3))
                )
                iterations.append(iteration_data)
                current_stage_hpwl = iteration_data.hpwl
                continue

        # Parse timing information - avoid duplicates by keeping the latest value for each stage
        stage_times = {}
        for pattern, stage_name in self.timing_patterns:
            matches = pattern.findall(log_text)
            if matches:
                # Keep the last occurrence (usually the most accurate)
                stage_times[stage_name] = float(matches[-1])

        # Convert to simplified StagePerformance objects
        for stage_name, duration in stage_times.items():
            stage_performance.append(StagePerformance(stage_name=stage_name, duration=duration))

        # Parse final results
        for line in lines:
            match = self.final_results_pattern.search(line)
            if match:
                final_iteration = int(match.group(1))
                final_hpwl = float(match.group(2))
                success = True
                break

        # Analyze convergence
        convergence_analysis = self.analyze_convergence(iterations) if iterations else None

        # Calculate total time
        total_time = 0
        for stage in stage_performance:
            if stage.stage_name == "total_placement":
                total_time = stage.duration
                break

        return PlacementResult(
            final_hpwl=final_hpwl or (iterations[-1].hpwl if iterations else 0),
            final_iteration=final_iteration or (iterations[-1].iteration if iterations else 0),
            total_time=total_time,
            success=success,
            error_message=error_message,
            iterations=iterations,
            stage_performance=stage_performance,
            convergence_analysis=convergence_analysis
        )

    def analyze_convergence(self, iterations: List[IterationMetrics]) -> ConvergenceAnalysis:
        """Analyze convergence patterns from iteration data"""
        if len(iterations) < 10:
            return ConvergenceAnalysis(
                converged=False,
                convergence_rate="unknown",
                oscillation_detected=False,
                early_convergence=False,
                final_iterations_stable=False,
                overflow_reduction_rate=0.0,
                hpwl_reduction_rate=0.0,
                total_iterations=len(iterations),
                overflow_converged=False,
                density_stabilized=False
            )

        # Separate global placement and other iterations
        gp_iterations = [it for it in iterations if it.detailed_step is not None]
        other_iterations = [it for it in iterations if it.detailed_step is None]

        # Analyze global placement convergence
        if len(gp_iterations) >= 2:
            first_hpwl = gp_iterations[0].hpwl
            last_gp_hpwl = gp_iterations[-1].hpwl
            gp_reduction = (first_hpwl - last_gp_hpwl) / first_hpwl if first_hpwl > 0 else 0

            # Overflow convergence
            first_overflow = gp_iterations[0].overflow if gp_iterations[0].overflow else 0
            last_overflow = gp_iterations[-1].overflow if gp_iterations[-1].overflow else 0
            overflow_converged = last_overflow < 0.1

            overflow_reduction = 0
            if first_overflow > 0:
                overflow_reduction = (first_overflow - last_overflow) / first_overflow
        else:
            gp_reduction = 0
            overflow_reduction = 0
            overflow_converged = False

        # Overall HPWL reduction
        if len(iterations) >= 2:
            overall_first_hpwl = iterations[0].hpwl
            overall_last_hpwl = iterations[-1].hpwl
            overall_reduction = (overall_first_hpwl - overall_last_hpwl) / overall_first_hpwl if overall_first_hpwl > 0 else 0
        else:
            overall_reduction = 0

        # Detect oscillation in recent iterations
        recent_iterations = iterations[-20:] if len(iterations) >= 20 else iterations
        hpwl_values = [it.hpwl for it in recent_iterations if it.hpwl is not None]
        oscillation = self.detect_oscillation(hpwl_values)

        # Determine convergence rate
        if gp_reduction > 0.15:
            convergence_rate = "fast"
        elif gp_reduction > 0.08:
            convergence_rate = "medium"
        else:
            convergence_rate = "slow"

        # Check if final iterations are stable
        stable = self.check_stability(hpwl_values[-10:]) if len(hpwl_values) >= 10 else False

        # Check density stabilization
        density_stabilized = False
        if len(gp_iterations) >= 10:
            recent_max_densities = [it.max_density for it in gp_iterations[-10:] if it.max_density is not None]
            if recent_max_densities:
                density_variance = max(recent_max_densities) - min(recent_max_densities)
                density_stabilized = density_variance / max(recent_max_densities) < 0.1

        return ConvergenceAnalysis(
            converged=gp_reduction > 0.02,
            convergence_rate=convergence_rate,
            oscillation_detected=oscillation,
            early_convergence=False,  # TODO: Implement more sophisticated early convergence detection
            final_iterations_stable=stable,
            overflow_reduction_rate=overflow_reduction,
            hpwl_reduction_rate=gp_reduction,
            total_iterations=len(iterations),
            overflow_converged=overflow_converged,
            density_stabilized=density_stabilized
        )

    def detect_oscillation(self, values: List[float], threshold: float = 0.02) -> bool:
        """Detect if values are oscillating"""
        if len(values) < 6:
            return False

        # Simple oscillation detection: check for alternating increases/decreases
        changes = []
        for i in range(1, len(values)):
            change = (values[i] - values[i-1]) / values[i-1] if values[i-1] != 0 else 0
            changes.append(change)

        # Count sign changes
        sign_changes = 0
        for i in range(1, len(changes)):
            if changes[i] * changes[i-1] < 0:  # Different signs
                sign_changes += 1

        return sign_changes > len(changes) * 0.4  # 40% of changes are sign changes

    def check_stability(self, values: List[float], threshold: float = 0.01) -> bool:
        """Check if values are stable (small variations)"""
        if len(values) < 2:
            return False

        mean_val = sum(values) / len(values)
        if mean_val == 0:
            return True

        max_deviation = max(abs(v - mean_val) for v in values) / mean_val
        return max_deviation < threshold

    def extract_trace(self, iterations: List[IterationMetrics], sample_rate: int = 150) -> List[Dict[str, Any]]:
        """Extract optimization trace with sampling"""
        if not iterations:
            return []

        trace = []
        # Sample fewer points for compact output
        for i, it in enumerate(iterations):
            if i % sample_rate == 0 or i == len(iterations) - 1:  # Include last iteration
                trace_data = {
                    "iteration": it.iteration,
                    "hpwl": it.hpwl
                }
                # Only include non-null fields
                if it.overflow is not None:
                    trace_data["overflow"] = it.overflow
                if it.max_density is not None:
                    trace_data["max_density"] = it.max_density
                if it.density_weight is not None:
                    trace_data["density_weight"] = it.density_weight
                if it.gamma is not None:
                    trace_data["gamma"] = it.gamma
                trace.append(trace_data)

        return trace


class DREAMPlaceEvaluator:
    """Main evaluator for DREAMPlace placement"""

    def __init__(self, dreamplace_path: str = "./"):
        self.dreamplace_path = Path(dreamplace_path)
        self.placer_script = self.dreamplace_path / "dreamplace" / "Placer.py"
        self.parser = DREAMPlaceLogParser()

    def evaluate(self, params: Dict[str, Any], config_file: str,
                 return_detailed_log: bool = True, timeout: int = 3600, sample_rate: int = 200) -> Dict[str, Any]:
        """
        Evaluate placement with given parameters

        Args:
            params: Dictionary of placement parameters
            config_file: Path to DREAMPlace configuration file
            return_detailed_log: Whether to return detailed analysis
            timeout: Timeout in seconds

        Returns:
            Dictionary with evaluation results
        """
        # Create temporary config file with modified parameters
        temp_config = self.create_temp_config(config_file, params)

        try:
            # Execute DREAMPlace
            start_time = time.time()
            log_text, success = self.run_placement(temp_config, timeout)
            execution_time = time.time() - start_time

            if not success:
                return {
                    "success": False,
                    "error_message": log_text,  # error message in log_text
                    "execution_time": execution_time,
                    "config_file": config_file
                }

            # Parse results
            result = self.parser.parse_log(log_text)
            result.success = True
            result.total_time = execution_time

            # Return structured results
            if return_detailed_log:
                return {
                    "success": True,
                    "final_hpwl": result.final_hpwl,
                    "final_iteration": result.final_iteration,
                    "execution_time": result.total_time,
                    "convergence_analysis": asdict(result.convergence_analysis) if result.convergence_analysis else None,
                    "stage_performance": [asdict(s) for s in result.stage_performance] if result.stage_performance else None,
                    "optimization_trace": self.parser.extract_trace(result.iterations, sample_rate=sample_rate),
                    "hpwl_progression": self.extract_hpwl_progression(result.iterations),
                    "overflow_progression": self.extract_overflow_progression(result.iterations),
                    "config_file": config_file
                }
            else:
                return {
                    "success": True,
                    "final_hpwl": result.final_hpwl,
                    "final_iteration": result.final_iteration,
                    "execution_time": result.total_time,
                    "config_file": config_file
                }

        except Exception as e:
            return {
                "success": False,
                "error_message": str(e),
                "execution_time": time.time() - start_time,
                "config_file": config_file
            }
        finally:
            # Keep temporary files for debugging
            # if 'temp_config' in locals() and os.path.exists(temp_config):
            #     os.unlink(temp_config)
            pass

    def create_temp_config(self, base_config: str, params: Dict[str, Any]) -> str:
        """Create temporary config file with modified parameters"""
        # Load base config
        with open(base_config, 'r') as f:
            config_data = json.load(f)

        # Apply parameters with special handling for global_place_stages
        both_levels = ['num_bins_x', 'num_bins_y']
        stage_only = ['wirelength', 'optimizer', 'learning_rate', 'learning_rate_decay']

        for key, value in params.items():
            if key in both_levels:
                # Set both top level and global_place_stages[0]
                config_data[key] = value
                config_data['global_place_stages'][0][key] = value
            elif key in stage_only:
                # Set only global_place_stages[0]
                config_data['global_place_stages'][0][key] = value
            else:
                # Set top level
                config_data[key] = value

        # Create temporary file in specific directory for debugging
        tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.json',
            prefix='dreamplace_eval_',
            dir=tmp_dir
        )
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(config_data, f, indent=2)

        return temp_path

    def run_placement(self, config_file: str, timeout: int) -> Tuple[str, bool]:
        """Execute DREAMPlace placer and return log"""
        cmd = ["python3", str(self.placer_script), config_file]

        try:
            # Run DREAMPlace
            result = subprocess.run(
                cmd,
                cwd=str(self.dreamplace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )

            return result.stdout + result.stderr, True

        except subprocess.TimeoutExpired:
            return f"Placement timed out after {timeout} seconds", False
        except subprocess.CalledProcessError as e:
            return f"Placement failed with error:\n{e.stderr}", False
        except Exception as e:
            return f"Unexpected error: {str(e)}", False

    def extract_hpwl_progression(self, iterations: List[IterationMetrics]) -> List[Dict[str, Any]]:
        """Extract HPWL progression at key points"""
        if not iterations:
            return []

        hpwl_progression = []

        # Add initial point
        if iterations[0].hpwl is not None:
            hpwl_progression.append({
                "iteration": 0,
                "hpwl": iterations[0].hpwl,
                "stage": "initial"
            })

        # Find key iterations (25%, 50%, 75% of global placement)
        gp_iterations = [it for it in iterations if it.detailed_step is not None]
        if len(gp_iterations) > 4:
            milestones = [0, len(gp_iterations)//4, len(gp_iterations)//2, 3*len(gp_iterations)//4, -1]
            for milestone in milestones:
                it = gp_iterations[milestone]
                hpwl_progression.append({
                    "iteration": it.iteration,
                    "hpwl": it.hpwl,
                    "stage": "global_placement"
                })

        # Add legalization result
        for it in iterations:
            if it.iteration > (gp_iterations[-1].iteration if gp_iterations else 0) and it.hpwl is not None:
                hpwl_progression.append({
                    "iteration": it.iteration,
                    "hpwl": it.hpwl,
                    "stage": "legalization"
                })
                break

        # Add final result
        if iterations[-1].hpwl is not None:
            hpwl_progression.append({
                "iteration": iterations[-1].iteration,
                "hpwl": iterations[-1].hpwl,
                "stage": "final"
            })

        return hpwl_progression

    def extract_overflow_progression(self, iterations: List[IterationMetrics]) -> List[Dict[str, Any]]:
        """Extract overflow progression (only for global placement)"""
        if not iterations:
            return []

        overflow_progression = []
        gp_iterations = [it for it in iterations if it.detailed_step is not None and it.overflow is not None]

        # Sample every 50 iterations
        for i, it in enumerate(gp_iterations):
            if i % 50 == 0 or i == len(gp_iterations) - 1:
                overflow_progression.append({
                    "iteration": it.iteration,
                    "overflow": it.overflow
                })

        return overflow_progression


def main():
    """Command line interface for the evaluator"""
    import argparse

    parser = argparse.ArgumentParser(description="DREAMPlace Evaluator for LLM-based HPO")
    parser.add_argument("config_file", help="Path to DREAMPlace configuration file")
    parser.add_argument("--params", help="JSON string with parameter overrides")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--detailed", action="store_true", default=True, help="Return detailed analysis (default)")
    parser.add_argument("--simple", action="store_true", help="Return only basic metrics (overrides --detailed)")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--sample-rate", type=int, default=200, help="Sampling rate for optimization trace (default: 200)")

    args = parser.parse_args()

    # Handle conflicting options
    if args.simple:
        args.detailed = False

    # Parse parameters
    params = {}
    if args.params:
        params = json.loads(args.params)

    # Create evaluator
    evaluator = DREAMPlaceEvaluator()

    # Run evaluation
    results = evaluator.evaluate(
        params=params,
        config_file=args.config_file,
        return_detailed_log=args.detailed,
        timeout=args.timeout,
        sample_rate=args.sample_rate
    )

    # If simple mode, extract only essential results
    if args.simple:
        simple_results = {
            "success": results["success"],
            "final_hpwl": results.get("final_hpwl"),
            "final_iteration": results.get("final_iteration"),
            "execution_time": results.get("execution_time"),
            "config_file": results["config_file"]
        }
        if not results["success"]:
            simple_results["error_message"] = results.get("error_message")
        results = simple_results

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
