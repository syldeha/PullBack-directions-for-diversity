"""Method parameters shared by the inpainting notebook and long evaluation."""

from dataclasses import dataclass


@dataclass
class InpaintingConfig:
    num_particles: int = 4
    resolution: int = 512
    ddim_steps: int = 50
    eta: float = 0.0
    initial_seed: int = 4242
    noise_seed_base: int = 20800

    # CADS paper baseline.
    cads_noise_scale: float = 1.0
    cads_start: int = 900
    cads_end: int = 600
    cads_rescale_factor: float = 0.5
    cads_condition_seed: int = 999

    # Adaptive disjoint pullback.
    pullback_rank: int = 24
    pullback_basis_timestep: int = 500
    pullback_basis_seed: int = 515
    pullback_basis_iterations: int = 3
    pullback_rho: float = 1.25
    pullback_start: int = 999
    pullback_end: int = 500
    pullback_schedule_power: float = 2.0
    pullback_direction_seed: int = 777
    pullback_refreshes: int = 2
    pullback_intermediate_rank: int = 8
    pullback_intermediate_iterations: int = 1
    pullback_intermediate_seed: int = 1515
    pullback_transition_steps: int = 2
    pullback_anchor_particle: int = 0
    pullback_response_region: str = "edit_mask"  # "global" or "edit_mask"

    # Particle-specific rho selection at one early Tweedie probe.
    rho_star_probe_timestep: int = 699
    rho_star_candidate_rhos: tuple = (0.10, 0.125, 0.15, 0.175, 0.20, 0.25)
    rho_star_max_clip_drop: float = 0.35
    rho_star_search_strategy: str = "beam"
    rho_star_max_combinations: int = 5_000_000
    rho_star_beam_width: int = 4096
    rho_star_constraint_fallback: str = "minimum_selectable"

    # TPSO token optimization.
    tpso_kappa: float = 0.80
    tpso_sigma: float = 0.01
    tpso_diversity_weight: float = 1.0
    tpso_learning_rate: float = 1e-3
    tpso_max_steps: int = 200
    tpso_min_steps: int = 50
    tpso_patience: int = 15
    tpso_initial_std: float = 1e-4
    tpso_seed: int = 3407
    tpso_ratio: float = 0.5

    def validate(self):
        if self.num_particles < 2:
            raise ValueError("num_particles must be at least 2")
        if self.pullback_rank < self.num_particles:
            raise ValueError("pullback_rank must be at least num_particles")
        if self.pullback_intermediate_rank < self.num_particles:
            raise ValueError(
                "pullback_intermediate_rank must be at least num_particles"
            )
        if not self.pullback_start > self.pullback_end:
            raise ValueError("pullback_start must be greater than pullback_end")
        if not self.cads_start > self.cads_end:
            raise ValueError("cads_start must be greater than cads_end")
        if not 0.0 < self.tpso_ratio <= 1.0:
            raise ValueError("tpso_ratio must be in (0, 1]")
        if self.pullback_response_region not in {"global", "edit_mask"}:
            raise ValueError(
                "pullback_response_region must be 'global' or 'edit_mask'"
            )
        if not self.rho_star_candidate_rhos:
            raise ValueError("rho_star_candidate_rhos cannot be empty")
        if any(float(rho) <= 0.0 for rho in self.rho_star_candidate_rhos):
            raise ValueError("rho-star candidate values must be positive")
        if len(set(map(float, self.rho_star_candidate_rhos))) != len(
            self.rho_star_candidate_rhos
        ):
            raise ValueError("rho-star candidate values must be unique")
        if self.rho_star_max_clip_drop < 0.0:
            raise ValueError("rho_star_max_clip_drop must be non-negative")
        if self.rho_star_search_strategy not in {"auto", "exact", "beam"}:
            raise ValueError(
                "rho_star_search_strategy must be 'auto', 'exact', or 'beam'"
            )
        if self.rho_star_max_combinations < 1:
            raise ValueError("rho_star_max_combinations must be positive")
        if self.rho_star_beam_width < 1:
            raise ValueError("rho_star_beam_width must be positive")
        if self.rho_star_constraint_fallback not in {
            "clean",
            "minimum_selectable",
            "error",
        }:
            raise ValueError(
                "rho_star_constraint_fallback must be clean, "
                "minimum_selectable, or error"
            )
