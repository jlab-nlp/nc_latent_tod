from collections import defaultdict
from typing import TypeVar, List, DefaultDict, Generic

T = TypeVar('T')


class PredictionRecorder(Generic[T]):
    """
    Records predictions for use in subsequent turns
    """
    # we'll just track these in memory for now
    predictions: DefaultDict[str, List[T]]

    def __init__(self):
        self.predictions = defaultdict(list)

    def add_prediction(self, dialogue_id: str, turn_id: int, prediction: T) -> None:
        """
        Store a prediction

        :param dialogue_id: dialogue id of this prediction
        :param turn_id: turn id of this prediction
        :param prediction: predicted state to store
        :return: None
        """
        # maybe make this idempotent?
        assert len(self.predictions[dialogue_id]) == turn_id, "unexpected number of existing predictions"
        self.predictions[dialogue_id].append(prediction)
        assert self.predictions[dialogue_id][turn_id] == prediction

    def retrieve_previous_turn_prediction(self, dialogue_id: str, turn_id: int) -> List[T]:
        """
        Retrieve the history of predictions for the given dialogue up to the turn prior to the given turn
        id. Turn ids must start at 0. i.e. for a turn_id of 2, returns the predictions for turn 0, turn 1, as a list.

        :param dialogue_id: id of dialogue to get predictions for
        :param turn_id: turn id to get predictions up to
        :return: list of previous predictions
        """
        if dialogue_id not in self.predictions:
            raise ValueError(f"no stored predictions for dialogue={dialogue_id}")
        if len(self.predictions[dialogue_id]) < turn_id:
            raise ValueError(f"not enough stored predictions ({len(self.predictions[dialogue_id])}) "
                             f"for dialogue={dialogue_id}, turn={turn_id}")
        result: List[T] = self.predictions[dialogue_id][:turn_id]
        assert len(result) == turn_id
        return result
