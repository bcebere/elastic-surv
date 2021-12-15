import json
import math
from typing import Any, List, Set, Tuple

import numpy as np

import elastic_surv.models as models
from elastic_surv.dataset import ESDataset


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class HyperbandOptimizer:
    def __init__(
        self,
        seeds: list = [
            models.CoxPHModel,
            models.DeepHitModel,
            models.LogisticHazardModel,
        ],
        max_iter: int = 81,
        eta: int = 3,
        verbose: bool = False,
        output_epochs: int = 200,
    ) -> None:
        """
        HyperbandOptimizer helps you pick a good model for your dataset.

        Args:
            seeds: list. Baseline models. They must implement the ModelSkeleton interface
            max_iter: int. maximum iterations per configuration
            eta: int. defines configuration downsampling rate.
            verbose: bool. Enable debug logging.
        """
        self.seeds = seeds
        self.max_iter = max_iter
        self.eta = eta

        def logeta(x: Any) -> Any:
            return math.log(x) / math.log(self.eta)

        self.logeta = logeta
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.candidate = {
            "score": -np.inf,
            "model": "nop",
            "params": {},
        }
        self.output_epochs = output_epochs
        self.visited: Set[str] = set()

        self.model_best_score = {}
        self.verbose = verbose

        for seed in self.seeds:
            if not issubclass(seed, models.ModelSkeleton):
                raise ValueError(f"Invalid type for seed {seed}")
            self.model_best_score[seed.name()] = -np.inf

    def _hash_dict(self, name: str, dict_val: dict) -> str:
        return json.dumps(
            {"name": name, "val": dict_val}, sort_keys=True, cls=NpEncoder
        )

    def _sample_model(self, model: Any, n: int) -> list:
        name = model.name()

        hashed = self._hash_dict(name, {})
        result: List[Tuple] = []
        if hashed not in self.visited:
            self.visited.add(hashed)
            result.append((model, {}))
            n -= 1

        for i in range(n):
            params = model.sample_hyperparameters()
            hashed = self._hash_dict(name, params)

            if hashed in self.visited:
                continue

            self.visited.add(hashed)
            result.append((model, params))

        return result

    def _sample(self, n: int) -> list:
        results = []
        for seed in self.seeds:
            results.extend(self._sample_model(seed, n))
        return results

    def _eval_params(self, model_t: Any, dataset: ESDataset, **params: Any) -> float:
        model = model_t(in_features=dataset.features(), verbose=self.verbose, **params)

        model.train(dataset)
        ev_score = model.score(dataset)

        score = ev_score["c_index"] - ev_score["brier_score"]

        if score > self.candidate["score"]:
            self.candidate = {
                "score": score,
                "params": params,
                "model": model_t,
            }
            if score > self.model_best_score[model.name()]:
                self.model_best_score[model.name()] = score

        return score

    def select_model(self, dataset: ESDataset) -> Any:
        """
        Run the AutoML routine and select the best model.
        """
        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = self._sample(math.ceil(n / len(self.seeds)))

            for i in range(s + 1):  # changed from s + 1
                if len(T) == 0:
                    break

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = int(math.ceil(n * self.eta ** (-i)))
                n_iterations = r * self.eta ** (i)

                scores = []

                for model, model_params in T:
                    score = self._eval_params(
                        model,
                        dataset,
                        epochs=int(n_iterations),
                        **model_params,
                    )
                    scores.append(score)
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                saved = int(math.ceil(n_configs / self.eta))

                indices = np.argsort(scores)
                T = [T[i] for i in indices]
                T = T[-saved:]
                scores = [scores[i] for i in indices]
                scores = scores[-saved:]

        if self.verbose:
            print(
                f"      >>> best candidate {self.candidate['model']}: ({self.candidate['params']}) --- score : {self.candidate['score']}"
            )

        self.seeds = sorted(
            self.model_best_score, key=self.model_best_score.get, reverse=True  # type: ignore
        )[:2]

        self.candidate["params"]["epochs"] = self.output_epochs

        return self.candidate["model"](
            in_features=dataset.features(), **self.candidate["params"]
        )
