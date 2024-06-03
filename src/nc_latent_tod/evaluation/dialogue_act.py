import json
from collections import Counter
from typing import List, Literal, Dict, Set, Callable

import fuzzywuzzy.fuzz

from nc_latent_tod.acts.act import Act, Entity

F1Term = Literal["TP", "FP", "FN", "TN"]
F1Metric = Literal["precision", "recall", "f1"]


def _get_act_names(acts: List[Act]) -> Set[str]:
    """
    Return a set of act names from a list of acts
    """
    return set(act.__class__.__name__ for act in acts)


def _get_slot_names(acts: List[Act]) -> Set[str]:
    """
    Return a set of slot names used in the acts in this list of acts
    """
    slots: Set[str] = set()
    for act in acts:
        for slot, value in vars(act).items():
            if isinstance(value, Entity):
                # we'll only check one nesting level
                slots.update(vars(value))
            else:
                slots.add(slot)
    return slots


def _get_slot_values(acts: List[Act]) -> Set[str]:
    """
    Return a set of slot values used in the acts in this list of acts
    """
    # json.dumps() to handle list type values
    values: Set[str] = set()
    for act in acts:
        for slot, value in vars(act).items():
            if isinstance(value, Entity):
                # we'll only check one nesting level
                values.update(json.dumps(v) for v in vars(value).values())
            else:
                values.add(json.dumps(value))
    return values


def _evaluate_tp_fp_fn(*, prediction: List[Act], gold: List[Act],
                       extract_values: Callable[[List[Act]], Set[str]] = _get_act_names,
                       fuzzy_ratio: int = 90) -> Dict[F1Term, int]:
    """
    Give an F1 score between two lists of dialogue acts, ignoring slots and values
    """
    predicted_act_classes: Set[str] = extract_values(prediction)
    # convert pred to gold according to fuzzy match:

    gold_act_classes: Set[str] = extract_values(gold)
    processed_pred_classes: Set[str] = set()
    for pred in predicted_act_classes:
        fuzzy_matched: bool = False
        for gold in gold_act_classes:
            if fuzzywuzzy.fuzz.partial_ratio(pred, gold) >= fuzzy_ratio:
                processed_pred_classes.add(gold)  # if match, add the match so it counts as a TP
                fuzzy_matched = True
                break
        if not fuzzy_matched:
            processed_pred_classes.add(pred)  # if no match, add the original pred

    # TP, FP, FN, TN
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    tp: int = len(processed_pred_classes.intersection(gold_act_classes))
    fp: int = len(processed_pred_classes.difference(gold_act_classes))
    fn: int = len(gold_act_classes.difference(processed_pred_classes))
    return {"TP": tp, "FP": fp, "FN": fn}


class ActPredictionEvaluator:
    """
    Evaluates dialogue act predictions, and can report metrics at any point in the evaluation process
    """

    acts_only: Counter[F1Term]
    fuzzy_ratio: int

    def __init__(self, fuzzy_ratio: int = 90):
        self.acts_only: Counter[F1Term] = Counter()
        self.slots_only: Counter[F1Term] = Counter()
        self.values_only: Counter[F1Term] = Counter()
        self.fuzzy_ratio = fuzzy_ratio

    def add_turn(self, *, pred_turn_acts: List[Act], gold_turn_acts: List[Act]) -> None:
        """
        Add a list of turns to the evaluation, and return the current TP/FP/FN counts
        """
        self.acts_only.update(_evaluate_tp_fp_fn(prediction=pred_turn_acts, gold=gold_turn_acts,
                                                 extract_values=_get_act_names, fuzzy_ratio=self.fuzzy_ratio))
        self.slots_only.update(_evaluate_tp_fp_fn(prediction=pred_turn_acts, gold=gold_turn_acts,
                                                  extract_values=_get_slot_names, fuzzy_ratio=self.fuzzy_ratio))
        self.values_only.update(_evaluate_tp_fp_fn(prediction=pred_turn_acts, gold=gold_turn_acts,
                                                   extract_values=_get_slot_values, fuzzy_ratio=self.fuzzy_ratio))

    def current_scores(self) -> Dict[str, float]:
        """
        Return the current F1 score, precision, and recall across acts/slots/values (independently, not grouped!)
        """
        result: Dict[str, float] = {}
        for prefix, value_extractor, counter in [
            ("acts", _get_act_names, self.acts_only),
            ("slots", _get_slot_names, self.slots_only),
            ("values", _get_slot_values, self.values_only)
        ]:
            precision: float = (counter["TP"] / (counter["TP"] + counter["FP"])) if (counter["TP"] + counter["FP"]) > 0 else 0
            recall: float = (counter["TP"] / (counter["TP"] + counter["FN"])) if (counter["TP"] + counter["FN"]) > 0 else 0
            f1: float = (2 * precision * recall / (precision + recall)) if (precision + recall > 0) else 0
            result.update({f"{prefix}_precision": precision, f"{prefix}_recall": recall, f"{prefix}_f1": f1})
        return result

    @staticmethod
    def evaluate_turn(*, pred_turn_acts: List[Act], gold_turn_acts: List[Act]) -> Dict[str, float]:
        """
        Evaluate a single turn of dialogue acts, returning the F1 score, precision, and recall, considering the act-type
        for each act in the turn (ignoring slots and values)
        """
        result: Dict[str, float] = {}
        for prefix, value_extractor in [
            ("acts", _get_act_names), ("slots", _get_slot_names), ("values", _get_slot_values)
        ]:
            scores: Dict[F1Term, float] = _evaluate_tp_fp_fn(prediction=pred_turn_acts, gold=gold_turn_acts,
                                                             extract_values=value_extractor)
            tp, fp, fn = scores["TP"], scores["FP"], scores["FN"]
            precision: float = (tp / (tp + fp)) if (tp + fp > 0) else 0
            recall: float = (tp / (tp + fn)) if (tp + fn > 0) else 0
            f1: float = (2 * precision * recall / (precision + recall)) if (precision + recall > 0) else 0
            result.update({f"{prefix}_precision": precision, f"{prefix}_recall": recall, f"{prefix}_f1": f1})
        return result
