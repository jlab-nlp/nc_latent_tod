from typing import List, TypedDict, Dict, NamedTuple, Literal, Tuple, Union, Optional, Type

from typing_extensions import NotRequired

from nc_latent_tod.acts.act import Act
from nc_latent_tod.db.types import DBEntity

RawMultiWOZDialogueAct = str
RawMultiWOZDialogueActListItem = str

RawMultiWOZGoalTopic = Literal['taxi', 'police', 'restaurant', 'hospital', 'hotel', 'general', 'attraction', 'train',
                               'booking']


class RawMultiWOZGoalItem(TypedDict):
    # slot short names and values, + invalid markers
    book: Optional[Dict[Union[str, Literal["invalid", "pre_invalid"]], Union[str, bool]]]
    fail_info: Optional[Dict[str, str]]  # slot short names and values
    info: Dict[str, str]  # slot short names and values
    fail_book: Optional[Dict[str, str]]  # slot short names and values
    reqt: Optional[List[str]]  # slots to request in return


# a flat dictionary from a service's informable slots to the string values that could be used in evaluation
ServiceBeliefState = Dict[str, str]

# a flat dictionary from a schema's informable slots to the string values that could be used in evaluation
SchemaBeliefState: Type[dict[str, dict[str, str]]] = Dict[str, ServiceBeliefState]


class RawMultiWOZGoal(TypedDict):
    """
    Data type for 'goal' key in raw MultiWOZ data: example: data.json['SNG01856.json']['goal']
    """
    message: List[str]
    taxi: RawMultiWOZGoalItem
    police: RawMultiWOZGoalItem
    hotel: RawMultiWOZGoalItem
    hospital: RawMultiWOZGoalItem
    train: RawMultiWOZGoalItem
    attraction: RawMultiWOZGoalItem
    restaurant: RawMultiWOZGoalItem
    topic: Dict[RawMultiWOZGoalTopic, bool]


BookedItem = Dict[str, str]
RawMultiWOZBeliefStateBook = Dict[str, Union[str, List[BookedItem]]]  # booked items under 'booked', rest are slots
RawMultiWOZBeliefStateSemi = Dict[str, str]  # allways slots and values


class RawMultiWOZDomainBeliefState(TypedDict):
    book: RawMultiWOZBeliefStateBook
    semi: RawMultiWOZBeliefStateSemi


class RawMultiWOZBeliefState(TypedDict):
    attraction: RawMultiWOZDomainBeliefState
    hotel: RawMultiWOZDomainBeliefState
    hospital: RawMultiWOZDomainBeliefState
    train: RawMultiWOZDomainBeliefState
    taxi: RawMultiWOZDomainBeliefState
    restaurant: RawMultiWOZDomainBeliefState
    police: RawMultiWOZDomainBeliefState


class RawMultiWOZLog(TypedDict):
    text: str
    metadata: RawMultiWOZBeliefState


class RawMultiWOZDialogue(TypedDict):
    goal: RawMultiWOZGoal
    log: List[RawMultiWOZLog]


class RawActSlotValuePair(NamedTuple):
    slot_name: str
    value: str


# indexes from dialogue id (without .json suffix) to str(turn_id) to dialogue act str and list
RawDialogueActJsonIndex = Dict[str, Dict[str, Dict[RawMultiWOZDialogueAct, List[RawActSlotValuePair]]]]

RawMultiWOZDomain = Literal["attraction", "hotel", "taxi", "restaurant", "police", "train", "hospital"]

SlotValue = str

# SlotNames are complete names including a domain and a slot, separated by a dash. e.g. "hotel-area"
SlotName = Literal["attraction-area", "attraction-name", "attraction-type", "bus-day", "bus-departure",
                   "bus-destination", "bus-leaveat", "hospital-department", "hotel-area", "hotel-book day",
                   "hotel-book people", "hotel-book stay", "hotel-internet", "hotel-name", "hotel-parking",
                   "hotel-pricerange", "hotel-stars", "hotel-type", "restaurant-area", "restaurant-book day",
                   "restaurant-book people", "restaurant-book time", "restaurant-food", "restaurant-name",
                   "restaurant-pricerange", "taxi-arriveby", "taxi-departure", "taxi-destination", "taxi-leaveat",
                   "train-arriveby", "train-book people", "train-day", "train-departure", "train-destination",
                   "train-leaveat"]

MultiWOZBeliefState = List[Tuple[SlotName, SlotValue]]


class MultiWOZTurn(TypedDict):
    dialogue_id: str
    turn_id: int
    user: str
    system_response: str
    history: List[str]
    system_acts: Dict[RawMultiWOZDialogueAct, List[RawActSlotValuePair]]
    belief_state: MultiWOZBeliefState
    prev_belief_state: MultiWOZBeliefState
    belief_state_delta: MultiWOZBeliefState
    degenerate_user: bool


class SystemAct(TypedDict):
    act: str
    service: str
    slot_name: NotRequired[str]
    value: NotRequired[str]


class SystemActBatch(TypedDict):
    act: List[str]
    service: List[str]
    slot_name: List[NotRequired[str]]
    value: List[NotRequired[str]]


# we'll support a few methods of passing around dialogue acts, though the primary should be List[Act], where List[str]
# is a list of serialized Act.to_json() strings
ValidActsRepresentation = Union[SystemActBatch, List[str], List[Act]]


class DatasetTurn(TypedDict):
    dialogue_id: str
    turn_id: int
    domains: List[str]
    # system always goes first! make sure for user init dialogues to insert an empty string system utterance. As such
    # system_utterances and user_utterances should always be the same length for all turns, regardless of initiative
    system_utterances: List[str]
    user_utterances: List[str]
    slot_values: SchemaBeliefState
    turn_slot_values: SchemaBeliefState
    last_slot_values: SchemaBeliefState
    last_system_response_acts: ValidActsRepresentation
    system_response_acts: ValidActsRepresentation
    system_response: str


SchemaBeliefStateTurnKey = Literal['slot_values', 'turn_slot_values', 'last_slot_values']


class DatasetTurnBatch(TypedDict):
    dialogue_id: List[str]
    turn_id: List[int]
    system_utterances: List[List[str]]
    user_utterances: List[List[str]]
    slot_values: List[SchemaBeliefState]
    turn_slot_values: List[SchemaBeliefState]
    last_slot_values: List[SchemaBeliefState]


class DatasetTurnLog(DatasetTurn):
    """
    Additional fields that can be populated as the result of a DST run with this turn
    """
    pred_belief_state: Optional[SchemaBeliefState]
    pred_delta_slot_values: Optional[SchemaBeliefState]
    pred_prior_context: Optional[SchemaBeliefState]
    not_valid: Optional[int]
    pred_system_response_acts: Optional[ValidActsRepresentation]
    # The list of active services inferred from the current system act(s)
    # Unioning this with the keys in pred_belief_state yield a full prediction of the active services in a turn.
    pred_act_based_active_service_names: Optional[List[str]]
    pred_delex_system_response: Optional[str]
    # include useful evaluation metrics
    jga: Optional[float]
    turn_acc: Optional[float]


class DatasetTurnResponseGenLog(DatasetTurn):
    """
    Additional fields that can be populated as the result of a response-gen run with this turn
    """
    pred_system_response: Optional[str]
    pred_system_act: Optional[str]
    db_result: Optional[List[DBEntity]]
    pred_slot_values: Optional[SchemaBeliefState]
    pred_active_service_names: Optional[List[str]]


SlotValue = Union[str, int]  # Most commonly strings, but occasionally integers

# MultiWOZ Dict is the dictionary format for slot values as provided in the dataset. It is flattened, and denotes a
# dictionary that can be immediately evaluated using exact-match based metrics on keys and values. keys are in
# domain-slot form e.g. {"hotel-area": "centre", ...}
MultiWOZDict = Dict[SlotName, SlotValue]
