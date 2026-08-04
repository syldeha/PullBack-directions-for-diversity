"""Focused CPU checks for the internal rho-star experiment."""

from types import SimpleNamespace

import torch

from methods import rho_star


def test_constrained_joint_selection():
    probe = {
        "candidate_rhos": torch.tensor([0.0, 0.1, 0.2]),
        "guidance_alignment": torch.tensor([
            [1.0, 0.99, 0.80],
            [1.0, 0.99, 0.99],
        ]),
        "relative_response": torch.tensor([
            [0.0, 0.20, 0.20],
            [0.0, 0.20, 0.80],
        ]),
        "predicted_x0_cosine_drift": torch.zeros(2, 3),
        "predicted_x0": torch.tensor([
            [[[[1.0, 0.0]]], [[[1.0, 0.0]]], [[[0.0, 1.0]]]],
            [[[[1.0, 0.0]]], [[[-1.0, 0.0]]], [[[0.0, -1.0]]]],
        ]),
    }

    selection = rho_star.select_rho_combination(
        probe,
        min_guidance_cosine=0.95,
        max_relative_response=0.50,
    )

    assert selection["selected_indexes"] == [0, 1]
    assert torch.allclose(
        torch.tensor(selection["selected_rhos"]),
        torch.tensor([0.0, 0.1]),
    )
    assert abs(selection["clean_predicted_diversity"]) < 1e-7
    assert abs(selection["predicted_diversity"] - 2.0) < 1e-7
    assert selection["number_of_feasible_combinations"] == 4

    rows = rho_star.candidate_diagnostic_rows(probe, selection)
    assert len(rows) == 6
    assert sum(row["selected"] for row in rows) == 2
    assert not rows[2]["feasible"]
    assert not rows[5]["feasible"]

    forced_nonzero = rho_star.select_rho_combination(
        probe,
        min_guidance_cosine=0.95,
        max_relative_response=0.50,
        selectable_rhos=[0.1],
        search_strategy="beam",
        beam_width=4,
    )
    assert forced_nonzero["selected_indexes"] == [1, 1]
    assert forced_nonzero["search_strategy"] == "beam"
    assert forced_nonzero["constraint_fallback_particles"] == []

    minimum_fallback = rho_star.select_rho_combination(
        probe,
        min_guidance_cosine=0.999,
        max_relative_response=0.01,
        selectable_rhos=[0.1, 0.2],
        constraint_fallback="minimum_selectable",
    )
    assert minimum_fallback["selected_indexes"] == [1, 1]
    assert minimum_fallback["constraint_fallback_particles"] == [0, 1]
    assert torch.allclose(
        torch.tensor(minimum_fallback["constraint_fallback_rhos"]),
        torch.tensor([0.1, 0.1]),
    )
    fallback_rows = rho_star.candidate_diagnostic_rows(probe, minimum_fallback)
    selected_rows = [row for row in fallback_rows if row["selected"]]
    assert all(row["feasible"] for row in selected_rows)
    assert not any(row["passes_constraints"] for row in selected_rows)


def test_clip_constrained_dino_selection():
    probe = {
        "candidate_rhos": torch.tensor([0.0, 0.1, 0.2]),
        "guidance_alignment": torch.ones(2, 3),
        "relative_response": torch.zeros(2, 3),
        "predicted_x0_cosine_drift": torch.zeros(2, 3),
        "clip_scores": torch.tensor([
            [30.0, 29.9, 29.0],
            [30.0, 29.0, 30.1],
        ]),
        "dino_features": torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        ]),
    }

    selection = rho_star.select_rho_combination_clip_dino(
        probe,
        max_clip_drop=0.25,
        selectable_rhos=[0.1, 0.2],
    )

    assert selection["selected_indexes"] == [1, 2]
    assert torch.allclose(
        torch.tensor(selection["selected_rhos"]),
        torch.tensor([0.1, 0.2]),
    )
    assert abs(selection["dino_diversity"] - 2.0) < 1e-7
    assert abs(selection["clean_dino_diversity"]) < 1e-7
    assert abs(selection["selected_clip_change_min"] + 0.1) < 1e-5
    assert selection["constraint_fallback_particles"] == []

    rows = rho_star.candidate_diagnostic_rows(probe, selection)
    assert rows[1]["passes_constraints"]
    assert not rows[2]["passes_constraints"]
    assert rows[5]["clip_change_from_clean"] > 0.0
    assert "dino_distance_from_clean" in rows[5]


def test_attach_clip_dino_probe_features():
    class FakeMetrics:
        @staticmethod
        def clip_score(images, prompt):
            assert prompt == "test prompt"
            values = [20.0 + float(value) for value in images]
            return sum(values) / len(values), values

        @staticmethod
        def dino_features(images):
            return torch.tensor([
                [1.0, float(value) + 1.0] for value in images
            ])

    model = rho_star.model
    saved = {
        "device": model.device,
        "model_dtype": model.model_dtype,
        "decode_latents": model.decode_latents,
    }
    try:
        model.device = torch.device("cpu")
        model.model_dtype = torch.float32
        model.decode_latents = lambda batch: list(range(len(batch)))
        probe = {
            "predicted_x0": torch.zeros(2, 3, 1, 2, 2),
        }
        enriched = rho_star.add_clip_dino_probe_features(
            probe,
            prompt="test prompt",
            metrics_calculator=FakeMetrics(),
            decode_batch_size=2,
        )
        assert tuple(enriched["clip_scores"].shape) == (2, 3)
        assert tuple(enriched["dino_features"].shape) == (2, 3, 2)
        assert torch.allclose(
            enriched["dino_features"].norm(dim=2), torch.ones(2, 3)
        )
    finally:
        for name, value in saved.items():
            setattr(model, name, value)


class FakeScheduler:
    config = SimpleNamespace(prediction_type="epsilon")
    alphas_cumprod = torch.full((1000,), 0.75, dtype=torch.float32)

    @staticmethod
    def scale_model_input(latents, timestep):
        del timestep
        return latents


class FakeUNet:
    def __call__(self, latents, timestep, encoder_hidden_states):
        del timestep
        batch = latents.shape[0]
        condition = encoder_hidden_states.float().reshape(batch, -1)
        sample = condition[:, :4].reshape(batch, 1, 2, 2)
        return SimpleNamespace(sample=sample.to(latents.dtype))


def test_probe_clean_control_and_shapes():
    model = rho_star.model
    saved = {
        "device": model.device,
        "model_dtype": model.model_dtype,
        "scheduler": model.scheduler,
        "unet": model.unet,
        "unet_particle_batch_size": model.unet_particle_batch_size,
    }
    try:
        model.device = torch.device("cpu")
        model.model_dtype = torch.float32
        model.scheduler = FakeScheduler()
        model.unet = FakeUNet()
        model.unet_particle_batch_size = 2

        latents = torch.tensor([
            [[[0.2, -0.1], [0.3, 0.4]]],
            [[[-0.3, 0.5], [0.1, -0.2]]],
        ])
        positive = torch.tensor([[[1.0, 0.2], [0.3, 0.8]]])
        negative = torch.zeros_like(positive)
        directions = torch.tensor([
            [[1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ])

        probe = rho_star.probe_rho_candidates(
            latents=latents,
            timestep=torch.tensor(899),
            positive_condition=positive,
            negative_condition=negative,
            number_of_real_tokens=2,
            full_scale_directions=directions,
            candidate_rhos=[0.0, 0.1],
            guidance_scale=7.5,
            schedule_start=999,
            schedule_end=500,
            schedule_power=1.0,
        )

        assert tuple(probe["guidance_alignment"].shape) == (2, 2)
        assert tuple(probe["relative_response"].shape) == (2, 2)
        assert tuple(probe["predicted_x0"].shape) == (2, 2, 1, 2, 2)
        assert torch.allclose(
            probe["guidance_alignment"][:, 0], torch.ones(2), atol=1e-6
        )
        assert torch.allclose(
            probe["relative_response"][:, 0], torch.zeros(2), atol=1e-7
        )
        assert torch.all(probe["relative_response"][:, 1] > 0)
        expected_envelope = (899.0 - 500.0) / (999.0 - 500.0)
        assert abs(probe["schedule_envelope"] - expected_envelope) < 1e-12
    finally:
        for name, value in saved.items():
            setattr(model, name, value)


def main():
    test_constrained_joint_selection()
    test_clip_constrained_dino_selection()
    test_attach_clip_dino_probe_features()
    test_probe_clean_control_and_shapes()
    print("rho-star tests passed")


if __name__ == "__main__":
    main()
