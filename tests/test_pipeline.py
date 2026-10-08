"""Exercise actual notebook code on synthetic data, without OSF/CLIP downloads."""
from pathlib import Path
import ast
import json
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "things_eeg2_clip.ipynb"
CELLS = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]


def containing(fragment):
    return next(source for source in CELLS if fragment in source)


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_selection_and_normalization_exclude_held_out_data(self):
        rng = np.random.default_rng(0)
        # Five optimization concepts and 200 unseen validation concepts.
        labels = np.repeat(np.arange(205), 2)
        Xtr = rng.normal(size=(410, 7, 24)).astype(np.float32)
        Xte = rng.normal(size=(8, 7, 24)).astype(np.float32)
        env = dict(np=np, torch=torch, nn=nn, F=F, SEED=0, DEVICE="cpu", N_CH=7, EMB_DIM=16,
                   train_vocab=[f"train-{i}" for i in range(205)], test_vocab=[f"test-{i}" for i in range(4)],
                   tr_idx=labels, te_idx=np.repeat(np.arange(4), 2), Xtr=Xtr, Xte=Xte,
                   Ztr_text=F.normalize(torch.randn(205, 16), dim=-1),
                   Zte_text=F.normalize(torch.randn(4, 16), dim=-1))
        exec(containing("class EEGEncoder2"), env)
        expected = Xtr[env["tr_i"]].mean((0, 2), keepdims=True)
        np.testing.assert_allclose(env["_mu"], expected)
        self.assertFalse(set(env["keep_g"]) & set(env["val_g"]))
        self.assertEqual(len(env["val_g"]), 200)
        self.assertFalse(np.isin(labels[env["tr_i"]], env["val_g"]).any())
        with tempfile.TemporaryDirectory() as tmp:
            env.update(RESULTS=Path(tmp), SUB=1, gloss={}, prompt=lambda uid: f"a photo of a {uid}", json=json)
            training = containing("best, best_state").replace("EPOCHS, BS, logit_scale = 120, 1024", "EPOCHS, BS, logit_scale = 2, 4")
            exec(training, env)
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertIn(metrics["selected_epoch"], [1, 2])
            self.assertGreaterEqual(metrics["top5"], metrics["top1"])
            self.assertEqual(metrics["test_candidates"], 4)
            self.assertTrue((Path(tmp) / "zero_shot_model.pt").is_file())
            model = env["zero_shot_model"]
            self.assertIs(env["model"], model)
            with torch.no_grad():
                output = model(torch.from_numpy(Xte))
            self.assertEqual(tuple(output.shape), (8, 16))
            torch.testing.assert_close(output.norm(dim=-1), torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_batched_ranking_and_small_candidate_vocabulary(self):
        tree = ast.parse(containing("def score("))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score")
        env = dict(torch=torch, np=np)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "score", "exec"), env)
        prototypes = torch.eye(3)
        labels = np.array([0, 2, 1, 2])
        inputs = prototypes[labels]
        top1, top5 = env["score"](nn.Identity(), inputs, labels, prototypes, k=5, batch_size=2)
        self.assertEqual((top1, top5), (1.0, 1.0))
        wrong_labels = np.array([1, 0, 2, 0])
        self.assertEqual(env["score"](nn.Identity(), inputs, wrong_labels, prototypes, batch_size=3), (0.0, 1.0))

    def test_homonym_ids_preserved_and_optional_models_isolated(self):
        tree = ast.parse(containing("def to_uid("))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "to_uid")
        env = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "to_uid", "exec"), env)
        self.assertEqual(env["to_uid"]("00079_bat1"), "bat1")
        self.assertEqual(env["to_uid"]("00080_bat2"), "bat2")
        self.assertEqual(env["to_uid"]("00001_aircraft_carrier"), "aircraft_carrier")
        # Optional cells must never rebind the selected zero-shot model.
        for source in [containing("if RUN_WITHIN_CONCEPT:"), containing("MIN_ISLAND =")]:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    self.assertNotIn(node.id, ["model", "zero_shot_model"])


if __name__ == "__main__":
    unittest.main()
