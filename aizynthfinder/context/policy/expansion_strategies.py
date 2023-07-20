""" Module containing classes that implements different expansion policy strategies
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aizynthfinder.chem import TemplatedRetroReaction
from aizynthfinder.context.policy.utils import _make_fingerprint
from aizynthfinder.utils.exceptions import PolicyException
from aizynthfinder.utils.logging import logger
from aizynthfinder.utils.models import load_model

if TYPE_CHECKING:
    from aizynthfinder.chem import TreeMolecule
    from aizynthfinder.chem.reaction import RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import Any, List, Sequence, Tuple


class ExpansionStrategy(abc.ABC):
    """
    A base class for all expansion strategies.

    The strategy can be used by either calling the `get_actions` method
    of by calling the instantiated class with a list of molecule.

    .. code-block::

        expander = MyExpansionStrategy("dummy", config)
        actions, priors = expander.get_actions(molecules)
        actions, priors = expander(molecules)

    :param key: the key or label
    :param config: the configuration of the tree search
    """

    _required_kwargs: List[str] = []

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        if any(name not in kwargs for name in self._required_kwargs):
            raise PolicyException(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}"
            )
        self._config = config
        self._logger = logger()
        self.key = key

    def __call__(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        return self.get_actions(molecules)

    @abc.abstractmethod
    def get_actions(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules

        :param molecules: the molecules to consider
        :return: the actions and the priors of those actions
        """


class TemplateBasedExpansionStrategy(ExpansionStrategy):
    """
    A template-based expansion strategy that will return `TemplatedRetroReaction` objects upon expansion.

    :param key: the key or label
    :param config: the configuration of the tree search
    :param source: the source of the policy model
    :param templatefile: the path to a HDF5 file with the templates
    :raises PolicyException: if the length of the model output vector is not same as the number of templates
    """

    _required_kwargs = ["source", "templatefile"]

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)

        source = kwargs["source"]
        templatefile = kwargs["templatefile"]

        self._logger.info(
            f"Loading template-based expansion policy model from {source} to {self.key}"
        )
        self.model = load_model(source, self.key, self._config.use_remote_models)

        self._logger.info(f"Loading templates from {templatefile} to {self.key}")
        if templatefile.endswith(".csv.gz") or templatefile.endswith(".csv"):
            self.templates: pd.DataFrame = pd.read_csv(
                templatefile, index_col=0, sep="\t"
            )
        else:
            self.templates = pd.read_hdf(templatefile, "table")

        if hasattr(self.model, "output_size") and len(self.templates) != self.model.output_size:  # type: ignore
            raise PolicyException(
                f"The number of templates ({len(self.templates)}) does not agree with the "  # type: ignore
                f"output dimensions of the model ({self.model.output_size})"
            )
            
        #Add parsing of cofactors to numpy arrays (as strings...)
        if "cofactors" in self.templates.columns:
            all_cofactors = set()
            for _, row in self.templates.iterrows():
                cofactors = row["cofactors"]
                if cofactors =="0":
                    continue
                cofactors=cofactors.split(',')
                for cofactor in cofactors:
                    cofactor_name = cofactor[1:]
                    all_cofactors.add(cofactor_name)
                    
            all_cofactors = list(all_cofactors)
            n_cofactors = max(len(all_cofactors),1)
            for ind in self.templates.index:
                row = self.templates.iloc[ind]
                result = np.zeros(n_cofactors,dtype=int)
                cofactors = row["cofactors"]
                if cofactors =="0":
                    self.templates.at[ind,"cofactors"]=np.array2string(result)[1:-1]
                    continue
                cofactors=cofactors.split(',')
                for cofactor in cofactors:
                    cofactor_sign = cofactor[0]
                    cofactor_name = cofactor[1:]
                    cofactor_ind = all_cofactors.index(cofactor_name)
                    if cofactor_sign=="+":
                        result[cofactor_ind]+=1
                    elif cofactor_sign=="-":
                        result[cofactor_ind]-=1
                    else:
                        raise ValueError("Cofactors have to start with +/- sign!")
                    
                #Strip away the "[]" since of course fromstring is deprecated and fromiter can not handle it.
                self.templates.at[ind,"cofactors"]=np.array2string(result)[1:-1]

    # pylint: disable=R0914
    def get_actions(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies and given cutoffs

        :param molecules: the molecules to consider
        :return: the actions and the priors of those actions
        """

        possible_actions = []
        priors = []

        for mol in molecules:
            model = self.model
            templates = self.templates

            all_transforms_prop = self._predict(mol, model)
            probable_transforms_idx = self._cutoff_predictions(all_transforms_prop)
            possible_moves = templates.iloc[probable_transforms_idx]
            probs = all_transforms_prop[probable_transforms_idx]

            priors.extend(probs)
            for idx, (move_index, move) in enumerate(possible_moves.iterrows()):
                metadata = dict(move)
                del metadata[self._config.template_column]
                metadata["policy_probability"] = float(probs[idx].round(4))
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key
                metadata["template_code"] = move_index
                metadata["template"] = move[self._config.template_column]
                possible_actions.append(
                    TemplatedRetroReaction(
                        mol,
                        smarts=move[self._config.template_column],
                        metadata=metadata,
                        use_rdchiral=self._config.use_rdchiral,
                        cofactors= np.fromiter(move["cofactors"].split(), dtype=int) if "cofactors" in possible_moves.columns else None
                    )
                )
        return possible_actions, priors  # type: ignore

    def _cutoff_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Get the top transformations, by selecting those that have:
            * cumulative probability less than a threshold (cutoff_cumulative)
            * or at most N (cutoff_number)
        """
        sortidx = np.argsort(predictions)[::-1]
        cumsum: np.ndarray = np.cumsum(predictions[sortidx])
        if any(cumsum >= self._config.cutoff_cumulative):
            maxidx = int(np.argmin(cumsum < self._config.cutoff_cumulative))
        else:
            maxidx = len(cumsum)
        maxidx = min(maxidx, self._config.cutoff_number) or 1
        return sortidx[:maxidx]

    @staticmethod
    def _predict(mol: TreeMolecule, model: Any) -> np.ndarray:
        fp_arr = _make_fingerprint(mol, model)
        return np.array(model.predict(fp_arr)).flatten()
