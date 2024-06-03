import json
import re
import unittest
from json import JSONDecodeError
from typing import List, Union, get_args, Dict
from unittest.mock import Mock

from datasets import Dataset, load_dataset
from tqdm import tqdm

from nc_latent_tod.kwargs_prompt.simple_ft_prompt import SimpleFTKwargsPromptGenerator
from nc_latent_tod.acts.act import Act, Entity
from nc_latent_tod.acts.act_definitions import Offer, Request, Inform, Confirm, NotifySuccess, NotifyFailure, \
    RequestAlternatives
from nc_latent_tod.acts.utils import get_acts_from_system_acts, get_act_loading_context
from nc_latent_tod.data_types import ValidActsRepresentation, DatasetTurn, SchemaBeliefState
from nc_latent_tod.db.multiwoz_db import MultiWOZDB
from nc_latent_tod.db.types import DBEntity
from nc_latent_tod.experiments.batch_client_lm_module import BatchLMClientActTagModule, AbstractLMClientModule, \
    BatchLMClientDSTModule, BatchLMClientPolicyModule, BatchLMClientResponseGenModule
from nc_latent_tod.experiments.config import GenerationConfig
from nc_latent_tod.experiments.data_types import SchemaGuidedActTaggingInputs, GenericInputs, SchemaGuidedDSTInputs
from nc_latent_tod.kwargs_prompt.parsing import remove_duplicate_kwargs, split_on_classes, match_keyword_arguments
from nc_latent_tod.kwargs_prompt.prompt import KwargsPromptGenerator, promptify_db_result
from nc_latent_tod.ontology.multiwoz.ontology import MultiWOZOntology
from nc_latent_tod.prompting.abstract_prompt_generator import PromptMode
from nc_latent_tod.resources import read_json_resource, read_resource
from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.utils.dialogue_states import remove_blank_values, compute_delta
from nc_latent_tod.utils.testing import test_suite

# copied from a run:
DST_PREAMBLE: str = """from typing import Dict, List

from dialogue.act import Act
from dialogue.management import BeliefState, update_state, DB, Policy, Agent, Intent


class DialogueAgent(Agent):

    history: List[str]
    state: BeliefState
    db: DB
    policy: Policy

    def handle_turn(self, prev_belief_state: BeliefState, last_system_utterance: str, last_system_acts: List[Act],
                    user_utterance: str, user_intent: List[Intent], db_result: List[Dict[str, str]] = None,
                    system_acts: List[Act] = None, system_response: str = None):
        self.history.append(user_utterance)
        self.state = update_state(prev_belief_state, last_system_acts, user_intent)
        if not db_result:
            db_result = self.db.query(self.state)
        if not system_acts:
            system_acts = self.policy.next_act(self.state, db_result, history=self.history)
        if not system_response:
            system_response = self.generate_response(system_acts)
        self.history.append(system_response)
        return system_response, system_acts

    def no_change(self) -> Intent:
        \"\"\"
        user communicates no change in the dialogue state
        \"\"\"
        pass

    
    @abc.abstractmethod
    def find_hotel(self, price_range: str = None, type: str = None, parking: str = None, book_day: str = None, book_people: int = None, book_stay: int = None, stars: int = None, internet: str = None, name: str = None, area: str = None) -> Intent:
        \"\"\"
        search for a hotel to stay in
    
        Parameters:
        -----------
        price_range: str
            price budget of the hotel  ('expensive', 'cheap', 'moderate', 'dontcare')
        type: str
            what is the type of the hotel  ('guesthouse', 'hotel', 'dontcare')
        parking: str
            whether the hotel has parking  ('yes', 'no', 'dontcare')
        book_day: str
            day of the hotel booking  (e.g. 'monday', 'tuesday', 'wednesday', 'thursday', ...)
        book_people: int
            number of people for the hotel booking  (e.g. 1, 2, 3, 4, ...)
        book_stay: int
            length of stay at the hotel  (e.g. 1, 2, 3, 4, ...)
        stars: int
            star rating of the hotel  (e.g. 0, 1, 2, 3, ...)
        internet: str
            whether the hotel has internet  ('yes', 'no', 'dontcare')
        name: str
            name of the hotel
        area: str
            area or place of the hotel  (e.g. 'centre', 'east', 'north', 'south', ...)
        \"\"\"
        pass


    def book_hotel(self, **kwargs):
        return self.find_hotel(**kwargs)


    @abc.abstractmethod
    def find_train(self, arrive_by: str = None, departure: str = None, day: str = None, book_people: int = None, leave_at: str = None, destination: str = None) -> Intent:
        \"\"\"
        search for trains that take you places
    
        Parameters:
        -----------
        arrive_by: str
            arrival time of the train
        departure: str
            departure location of the train  (e.g. 'birmingham new street', 'bishops stortford', 'broxbourne', 'cambridge', ...)
        day: str
            day of the train  (e.g. 'monday', 'tuesday', 'wednesday', 'thursday', ...)
        book_people: int
            how many train tickets you need  (e.g. 0, 1, 2, 3, ...)
        leave_at: str
            leaving time for the train
        destination: str
            destination of the train  (e.g. 'birmingham new street', 'bishops stortford', 'broxbourne', 'cambridge', ...)
        \"\"\"
        pass


    def book_train(self, **kwargs):
        return self.find_train(**kwargs)


    @abc.abstractmethod
    def find_attraction(self, area: str = None, name: str = None, type: str = None) -> Intent:
        \"\"\"
        search for places to see for leisure
    
        Parameters:
        -----------
        area: str
            area to search for attractions  (e.g. 'centre', 'east', 'north', 'south', ...)
        name: str
            name of the attraction
        type: str
            type of the attraction  (e.g. 'architecture', 'boat', 'cinema', 'college', ...)
        \"\"\"
        pass


    @abc.abstractmethod
    def find_restaurant(self, price_range: str = None, area: str = None, food: str = None, name: str = None, book_day: str = None, book_people: int = None, book_time: str = None) -> Intent:
        \"\"\"
        search for places to wine and dine
    
        Parameters:
        -----------
        price_range: str
            price budget for the restaurant  ('cheap', 'expensive', 'moderate', 'dontcare')
        area: str
            area or place of the restaurant  (e.g. 'centre', 'east', 'north', 'south', ...)
        food: str
            the cuisine of the restaurant you are looking for
        name: str
            name of the restaurant
        book_day: str
            day of the restaurant booking  (e.g. 'monday', 'tuesday', 'wednesday', 'thursday', ...)
        book_people: int
            how many people for the restaurant reservation  (e.g. 1, 2, 3, 4, ...)
        book_time: str
            time of the restaurant booking
        \"\"\"
        pass


    def book_restaurant(self, **kwargs):
        return self.find_restaurant(**kwargs)


    @abc.abstractmethod
    def book_taxi(self, leave_at: str = None, destination: str = None, departure: str = None, arrive_by: str = None) -> Intent:
        \"\"\"
        book taxis to travel between places
    
        Parameters:
        -----------
        leave_at: str
            leaving time of taxi
        destination: str
            destination of taxi
        departure: str
            departure location of taxi
        arrive_by: str
            arrival time of taxi
        \"\"\"
        pass



if __name__ == '__main__':
    agent = DialogueAgent()

    # Provide the call matching the user's intent in this context
    # Format integers as `int`s, not strings. String values do not need underscores for spaces. Put time values in 'hh:mm' format"""

# copying here, these are AUTO-GENERATED in the real system, do not import elsewhere!
class Hotel(Entity):
    """
        hotel reservations and vacation stays

        Parameters:
        -----------
        price_range: str
            price budget of the hotel  (e.g. 'expensive', 'cheap', 'moderate', 'dontcare')
        type: str
            what is the type of the hotel  (e.g. 'guesthouse', 'hotel', 'dontcare')
        parking: str
            whether the hotel has parking  (e.g. 'yes', 'no', 'dontcare')
        book_day: str
            day of the hotel booking  (e.g. 'monday', 'tuesday', 'wednesday', 'thursday', ...)
        book_people: int
            number of people for the hotel booking  (e.g. '1', '2', '3', '4', ...)
        book_stay: int
            length of stay at the hotel  (e.g. '1', '2', '3', '4', ...)
        stars: int
            star rating of the hotel  (e.g. '0', '1', '2', '3', ...)
        internet: str
            whether the hotel has internet  (e.g. 'yes', 'no', 'dontcare')
        name: str
            name of the hotel
        area: str
            area or place of the hotel  (e.g. 'centre', 'east', 'north', 'south', ...)
        address: str
            address of the hotel
        phone: str
            phone number of the hotel
        postcode: str
            postal code of the hotel
        ref: str
            reference number of the hotel booking
        """
    price_range: str = None
    type: str = None
    parking: str = None
    book_day: str = None
    book_people: int = None
    book_stay: int = None
    stars: int = None
    internet: str = None
    name: str = None
    area: str = None
    address: str = None
    phone: str = None
    postcode: str = None
    ref: str = None


class Train(Entity):
    """
        find trains that take you to places

        Parameters:
        -----------
        arrive_by: str
            arrival time of the train
        departure: str
            departure location of the train  (e.g. 'birmingham new street', 'bishops stortford', 'broxbourne', 'cambridge', ...)
        day: str
            day of the train  (e.g. 'monday', 'tuesday', 'wednesday', 'thursday', ...)
        book_people: int
            how many train tickets you need  (e.g. '0', '1', '2', '3', ...)
        leave_at: str
            leaving time for the train
        destination: str
            destination of the train  (e.g. 'birmingham new street', 'bishops stortford', 'broxbourne', 'cambridge', ...)
        id: str
            id of the train
        ref: str
            reference number of the train booking
        price: str
            price of the train
        duration: str
            duration of the travel
        """
    arrive_by: str = None
    departure: str = None
    day: str = None
    book_people: int = None
    leave_at: str = None
    destination: str = None
    id: str = None
    ref: str = None
    price: str = None
    duration: str = None


class Attraction(Entity):
    """
        find touristy stuff to do around you

        Parameters:
        -----------
        area: str
            area to search for attractions  (e.g. 'centre', 'east', 'north', 'south', ...)
        name: str
            name of the attraction
        type: str
            type of the attraction  (e.g. 'architecture', 'boat', 'cinema', 'college', ...)
        entrance_fee: str
            how much is the entrance fee
        open_hours: str
            open hours of the attraction
        address: str
            address of the attraction
        phone: str
            phone number of the attraction
        postcode: str
            postal code of the attraction
        """
    area: str = None
    name: str = None
    type: str = None
    entrance_fee: str = None
    open_hours: str = None
    address: str = None
    phone: str = None
    postcode: str = None


class Restaurant(Entity):
    """
        find places to dine and whet your appetite

        Parameters:
        -----------
        price_range: str
            price budget for the restaurant  (e.g. 'cheap', 'expensive', 'moderate', 'dontcare')
        area: str
            area or place of the restaurant  (e.g. 'centre', 'east', 'north', 'south', ...)
        food: str
            the cuisine of the restaurant you are looking for
        name: str
            name of the restaurant
        book_day: str
            day of the restaurant booking  (e.g. 'monday', 'tuesday', 'wednesday', 'thursday', ...)
        book_people: int
            how many people for the restaurant reservation  (e.g. '1', '2', '3', '4', ...)
        book_time: str
            time of the restaurant booking
        address: str
            address of the restaurant
        phone: str
            phone number of the restaurant
        postcode: str
            postal code of the restaurant
        ref: str
            reference number of the restaurant booking
        """
    price_range: str = None
    area: str = None
    food: str = None
    name: str = None
    book_day: str = None
    book_people: int = None
    book_time: str = None
    address: str = None
    phone: str = None
    postcode: str = None
    ref: str = None


class Taxi(Entity):
    """
        rent cheap cabs to avoid traffic

        Parameters:
        -----------
        leave_at: str
            leaving time of taxi
        destination: str
            destination of taxi
        departure: str
            departure location of taxi
        arrive_by: str
            arrival time of taxi
        type: str
            car type of the taxi
        phone: str
            phone number of the taxi
        """
    leave_at: str = None
    destination: str = None
    departure: str = None
    arrive_by: str = None
    type: str = None
    phone: str = None


@test_suite("unit_build")
class KwargsTestCases(unittest.TestCase):
    dataset: Dataset
    pg: KwargsPromptGenerator
    db: MultiWOZDB
    simple_ft_pg: SimpleFTKwargsPromptGenerator

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        schema = read_multiwoz_schema()
        cls.schema = schema
        cls.act_loading_context = get_act_loading_context(schema)
        cls.dataset = load_dataset("Brendan/multiwoz_turns_v22", split="validation")
        db = MultiWOZDB()
        cls.db = db
        ontology = MultiWOZOntology.create_ontology()
        cls.pg = KwargsPromptGenerator(
            schema=schema,
            ontology=ontology,
            db=db
        )
        cls.simple_ft_pg = SimpleFTKwargsPromptGenerator(
            schema=schema,
            ontology=ontology,
            db=db
        )
        cls.one_turn = {
            'dialogue_id': 'MUL1600.json',
            'turn_id': 0,
            'domains': ['restaurant', 'train'],
            'system_utterances': [''],
            'user_utterances': ['I am looking for a train that leaves Cambridge going to Leicester arriving by 16:15'],
            'slot_values': {'train': {'arrive by': '16:15', 'departure': 'cambridge', 'destination': 'leicester'}},
            'turn_slot_values': {'train': {'arrive by': '16:15', 'departure': 'cambridge', 'destination': 'leicester'}},
            'last_slot_values': {},
            'last_system_response_acts': [],
            'system_response_acts': [
                '{"__type": "Inform", "entity": {"__type": "Train", "choice": "77"}}',
                '{"__type": "Request", "service": "train", "values": ["day"]}'
            ],
            'system_response': 'We have 77 options available. Is there a certain day you want to travel?'
        }

    def test_simple_get_acts_from_from_dict(self):
        # Simple Inform
        act_strs = [
            "{'service': 'restaurant', 'act': 'inform', 'slot_name': 'price', 'value': 'cheap'}",
            "{'service': 'restaurant', 'act': 'inform', 'slot_name': 'name', 'value': 'applebees'}"
        ]
        acts: List[Act] = get_acts_from_system_acts(act_strs, self.schema, act_loading_context=self.act_loading_context)
        self.assertListEqual(acts, [
            Inform(entity=Restaurant(price='cheap', name='applebees')),
        ])

        # Simple Request
        act_strs = [
            "{'service': 'restaurant', 'act': 'request', 'slot_name': 'price', 'value': '?'}",
            "{'service': 'restaurant', 'act': 'request', 'slot_name': 'area', 'value': '?'}",
            "{'service': 'restaurant', 'act': 'request', 'slot_name': 'food', 'value': '?'}",
        ]
        acts: List[Act] = get_acts_from_system_acts(act_strs, self.schema, act_loading_context=self.act_loading_context)
        self.assertListEqual(acts, [
            Request(service='restaurant', values=['price', 'area', 'food'])
        ])

        # Simple Recommend/Offer
        act_strs = [
            "{'service': 'hotel', 'act': 'recommend', 'slot_name': 'price', 'value': 'cheap'}",
            "{'service': 'hotel', 'act': 'recommend', 'slot_name': 'name', 'value': ' warkworth house'}"
        ]
        acts: List[Act] = get_acts_from_system_acts(act_strs, self.schema, act_loading_context=self.act_loading_context)
        self.assertListEqual(acts, [
            Offer(entity=Hotel(price='cheap', name='warkworth house')),
        ])

        # Simple Confirm (offerbook)
        act_strs = [
            "{'service': 'attraction', 'act': 'offerbook', 'slot_name': 'area', 'value': 'east'}",
            "{'service': 'attraction', 'act': 'offerbook', 'slot_name': 'type', 'value': 'museum'}"
        ]
        acts: List[Act] = get_acts_from_system_acts(act_strs, self.schema, act_loading_context=self.act_loading_context)
        self.assertListEqual(acts, [
            Confirm(entity=Attraction(area='east', type='museum')),
        ])

        # Simple NotifySuccess (offerbooked)
        act_strs = [
            "{'service': 'hotel', 'act': 'offerbooked', 'slot_name': 'none', 'value': 'none'}",
        ]
        acts: List[Act] = get_acts_from_system_acts(act_strs, self.schema, act_loading_context=self.act_loading_context)
        self.assertListEqual(acts, [
            NotifySuccess(service='hotel'),
        ])

        # Simple NotifyFailure (nooffer)
        act_strs = [
            "{'service': 'taxi', 'act': 'nooffer', 'slot_name': 'none', 'value': 'none'}",
        ]
        acts: List[Act] = get_acts_from_system_acts(act_strs, self.schema, act_loading_context=self.act_loading_context)
        self.assertListEqual(acts, [
            NotifyFailure(service='taxi'),
        ])

        # Simple RequestAlternatives (reqmore)
        act_strs = [
            "{'service': 'hotel', 'act': 'reqmore', 'slot_name': 'none', 'value': 'none'}",
        ]
        acts: List[Act] = get_acts_from_system_acts(act_strs, self.schema, act_loading_context=self.act_loading_context)
        self.assertListEqual(acts, [
            RequestAlternatives(),
        ])

    def test_unfill_act_values(self):
        acts = [
            Inform(entity=Restaurant(price='cheap', name='applebees')),
            Request(service='restaurant', values=['price', 'area', 'food']),
        ]
        act_jsons: List[str] = [act.to_json() for act in acts]
        self.assertListEqual(
            get_acts_from_system_acts(
                act_jsons, self.schema, unfill_act_values=True, act_loading_context=self.act_loading_context
            ),
            [
                Inform(entity=Restaurant(price='[value_price]', name='[value_name]')),
                Request(service='restaurant', values=['price', 'area', 'food'])
            ]
        )

    def test_get_acts_from_system_acts(self):
        # for every observed system act, make sure we can create our acts from it:
        for turn in tqdm(self.dataset, desc="validating transformation of system act annotations"):
            acts: ValidActsRepresentation = turn['system_response_acts']

            # validate that we can transform to list of Act
            our_acts: List[Act] = get_acts_from_system_acts(
                acts, self.schema, act_loading_context=self.act_loading_context
            )
            # acts could be blank, e.g. ['']
            if len(acts) >= 1 and any(a for a in acts):
                self.assertGreaterEqual(len(our_acts), 1)
                for act in our_acts:
                    self.assertIsNotNone(act)
                    self.assertIsInstance(act, Act)

                    # validate serialization/deserialization
                    act_class = type(act)
                    self.assertEqual(
                        act, act_class.from_dict(act.to_dict(), loading_context=globals())
                    )

                # validate calling again is idempotent (in case we're passed a list of acts, verfies our kwarg de-dupe
                # doesn't clobber good values)
                self.assertListEqual(our_acts, get_acts_from_system_acts(
                    our_acts, self.schema, act_loading_context=self.act_loading_context
                ))

                # validate that we can call with a list of json string (and dumps as well)
                json_acts: List[str] = [act.to_json() for act in our_acts]
                self.assertIsInstance(json_acts, list)
                self.assertIsInstance(json_acts[0], str)
                self.assertListEqual(our_acts, get_acts_from_system_acts(
                    json_acts, self.schema, act_loading_context=self.act_loading_context)
                                     )

    def test_acts_completion_parser(self):
        parse = self.pg.parse_sys_act_completion(
            "[Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3)),\n" +
            "Offer(entity=Hotel(name='acorn house', area='east', price_range='cheap', stars=4))")

        self.assertListEqual(parse,
                             [Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3)),
                              Offer(entity=Hotel(name='acorn house', area='east', price_range='cheap', stars=4))]
                             )
        acts: List[Act] = self.pg.parse_sys_act_completion("[Offer(agent=agent, entity=Hotel(area='west'))]")
        act_jsons: List[str] = [act.to_json() for act in acts]
        self.assertListEqual(act_jsons, [Offer(entity=Hotel(area='west')).to_json()])

        acts: List[Act] = self.pg.parse_sys_act_completion("[Offer(entity=Hotel(area='west', agent=agent))]")
        act_jsons: List[str] = [act.to_json() for act in acts]
        self.assertListEqual(act_jsons, [Offer(entity=Hotel(area="west")).to_json()])

        completion: str = "[Request(service='restaurant', values=['ref'])]\n" + \
                          "    ),\n    system_response=\"Would you like a reference number?\"\n" \
                          "    system_acts=[Request(service='restaurant', values=['ref'])]"
        print(completion)
        self.assertEqual(self.pg.parse_sys_act_completion(completion), [Request(service='restaurant', values=['ref'])])

    def test_match_keyword_arguments(self):
        string: str = "entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3, stars=3)"
        matches = match_keyword_arguments(string)
        self.assertListEqual(matches, [('name', "'allenbell'"), ('area', "'east'"), ('price_range', "'cheap'"),
                                       ('stars', '3'), ('stars', '3')])

    def test_remove_duplicate_kwargs(self):
        # Test 1: Basic test with string and numeric arguments.
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3, stars=3))"
            ),
            "Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3))",
        )
        # Test 2: All numeric arguments.
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(hotel=Hotel(stars=9_000, rating=5, rating=4, rating=3))"
            ),
            "Offer(hotel=Hotel(stars=9_000, rating=5))",
        )
        # Test 3: multi word arguments.
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(entity=Hotel(name='motel', type='a great hotel please', type='cheap is a maybe?', type='economy isn\'t good'))"
            ),
            "Offer(entity=Hotel(name='motel', type='a great hotel please'))",
        )
        # Test 4: Mixed quotes and no quotes. List handling also
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(hotel=Hotel(name='inn', type=['premium', 4, 2], type=['luxury']))"
            ),
            "Offer(hotel=Hotel(name='inn', type=['premium', 4, 2]))",
        )
        # Test 5: Duplicate at the start and the end.
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(entity=Restaurant(name='inn', name='lodge', type=premium))"
            ),
            "Offer(entity=Restaurant(name='inn', type=premium))",
        )
        # Test 6: Only one unique argument, rest are duplicates.
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(hotel=Hotel(name='inn', name='lodge', name='motel'))"
            ),
            "Offer(hotel=Hotel(name='inn'))",
        )
        # Test 7: No duplicates.
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(entity=Hotel(name='inn', type=premium))"
            ),
            "Offer(entity=Hotel(name='inn', type=premium))",
        )
        # Test 8: no args
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer()"
            ),
            "Offer()",
        )
        # Test 8: not as keyword
        self.assertEqual(
            remove_duplicate_kwargs(
                "Offer(Hotel(name='acorn guest house'))"
            ),
            "Offer(Hotel(name='acorn guest house'))"
        )
        # Test 9: requests
        self.assertEqual(
            remove_duplicate_kwargs("[Request(service='hotel', values=['area', 'price'])"),
            "[Request(service='hotel', values=['area', 'price'])"
        )
        # Test 10: multiple offers
        completion: str = "[Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3)),\n" + \
                          "Offer(entity=Hotel(name='acorn house', area='east', price_range='cheap', stars=4))"
        self.assertEqual(completion, remove_duplicate_kwargs(completion))
        # Test 10: multiple offers (w/ dupe)
        completion: str = "[Offer(entity=Hotel(name='allenbell', area='east', area='south', price_range='cheap', stars=3)),\n" + \
                          "Offer(entity=Hotel(name='acorn house', area='east', price_range='cheap', stars=4))"
        fixed_completion: str = "[Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3)),\n" + \
                                "Offer(entity=Hotel(name='acorn house', area='east', price_range='cheap', stars=4))"
        self.assertEqual(fixed_completion, remove_duplicate_kwargs(completion))

    def test_split_on_classes(self):
        completion: str = "[Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3)),\n" + \
                          "Offer(entity=Hotel(name='acorn house', area='east', price_range='cheap', stars=4))]"
        result = split_on_classes(completion)
        self.assertListEqual(result, [
            "[Offer(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3))",
            ",\nOffer(entity=Hotel(name='acorn house', area='east', price_range='cheap', stars=4))",
            "]"
        ])

    @unittest.skip
    def test_acts_assumption_none_values_are_boolean_yes(self):
        # turns with 'none' as a value and a slot_name of 'parking' or 'internet' are used to communicate the presence
        # of parking or internet. We want to convert these to 'yes' on read in, and need to make sure these are the only
        # such cases, so that we are not misinterpreting the 'none' for other values.
        for turn in tqdm(self.dataset, desc="validating system act assumptions around 'none' as a value"):
            for act in turn['system_response_acts']:
                try:
                    act_obj = json.loads(act)
                except JSONDecodeError as e:
                    act_obj = eval(act, {}, {})
                if act_obj['slot_name'] != 'none' and act_obj['value'] == 'none':
                    self.assertIn(act_obj['slot_name'], ('internet', 'parking'))
                    self.assertIn(act_obj['service'], ('hotel', 'restaurant', 'attraction', 'taxi', 'train'))

    def test_e2e_tautologies(self):
        valid_service_names = [svc['service_name'] for svc in self.schema]
        for turn in tqdm(self.dataset, desc="validating development set prompting & completion parsing"):
            turn: DatasetTurn
            dialogue_id: str = turn['dialogue_id']
            turn_id: int = turn['turn_id']
            # take turn and get expected update string
            prev_state: SchemaBeliefState = turn['last_slot_values']
            curr_state: SchemaBeliefState = turn['slot_values']
            prev_state = remove_blank_values({k: v for k, v in prev_state.items() if k in valid_service_names})
            curr_state = remove_blank_values({k: v for k, v in curr_state.items() if k in valid_service_names})
            delta = remove_blank_values({k: v for k, v in turn['turn_slot_values'].items() if k in valid_service_names})
            self.assertEqual(compute_delta(prev_state, curr_state), delta)
            turn_strings: List[str] = self.pg.get_turn_strings(turn)
            update_str = self.pg.get_user_intent_str(prev_state, curr_state, turn_strings)
            parse: SchemaBeliefState = self.pg.parse_dst_completion(update_str, prev_state,
                                                                    exceptions_are_empty=False)
            if (dialogue_id, turn_id) == ('MUL2618.json', 4):
                # this one gets matched up on a state reference, so patching it
                self.assertEqual(parse['taxi']['destination'], 'whale of time')
                parse['taxi']['destination'] = 'whale of a time'
            if (dialogue_id, turn_id) == ('PMUL2219.json', 7):
                # this one gets matched up on a state reference, so patching it
                self.assertEqual(parse['taxi']['destination'], 'pizza hut cherry hilton')
                parse['taxi']['destination'] = 'pizza hut cherry'
            self.assertEqual(parse, curr_state)

    def test_simple_serialization(self):
        acts: List[Act] = [
            Inform(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3)),
            RequestAlternatives(),
            Confirm(entity=Attraction(area='east', type='museum')),
        ]
        act_jsons: List[str] = [act.to_json() for act in acts]
        self.assertListEqual(
            get_acts_from_system_acts(act_jsons, self.schema, act_loading_context=self.act_loading_context),
            acts
        )
        self.assertListEqual(
            [
                act.to_json() for act in
                get_acts_from_system_acts(act_jsons, self.schema, act_loading_context=self.act_loading_context)
            ],
            act_jsons
        )

    def test_canonical_parse(self):
        invalid_dst_completion: str = "[agent.find_hotel(area='norwich', name='the oaklands hotel', stars=3)"
        self.assertEqual(
            # removes area
            "[agent.find_hotel(stars=3, name=\"the oaklands hotel\")]",
            self.pg.get_canonical_dst_completion(invalid_dst_completion, {}, [], "causal_dst")
        )

    def test_get_service_names_from_acts(self):
        acts: List[Act] = [
            Inform(entity=Hotel(name='allenbell', area='east', price_range='cheap', stars=3)),
            RequestAlternatives(),
            Confirm(entity=Attraction(area='east', type='museum'))
        ]
        self.assertListEqual(sorted(self.pg.get_service_names_from_acts(acts)), ['attraction', 'hotel'])

    def test_promptify_db_results(self):
        entities: List[DBEntity] = self.db.query(service_name='hotel', constraints={'area': 'east'})
        db_string: str = promptify_db_result(entities, service_name='hotel')
        self.assertEqual(db_string, "{'num_results': 7, 'values': [Hotel, Hotel, Hotel, ...]}")
        entities: List[DBEntity] = self.db.query(service_name='restaurant',
                                                 constraints={'area': 'east', 'food': 'chinese'})
        db_string: str = promptify_db_result(entities, service_name='restaurant')
        self.assertEqual(db_string, "{'num_results': 1, 'values': [Restaurant]}")
        entities: List[DBEntity] = self.db.query(service_name='restaurant', constraints={'name': 'name not found'})
        db_string: str = promptify_db_result(entities, service_name='restaurant')
        self.assertEqual(db_string, "{'num_results': 0, 'values': []}")

    def test_ft_prompt_dst(self):
        for turn in load_dataset("Brendan/multiwoz_turns_v22", split="train[0:20]"):
            ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(turn, mode="causal_dst")
            eval_prompt = self.simple_ft_pg.get_dst_prompt(
                turn_user_utterances=turn['user_utterances'],
                turn_system_utterances=turn['system_utterances'],
                belief_state_history=[turn['last_slot_values']],
                mode="causal_dst"
            )
            self.assertTrue(eval_prompt.startswith(ft_prompt))
            full_ft_text: str = ft_prompt + ft_completion
            self.assertTrue(full_ft_text.startswith(eval_prompt))

    def test_exact_ft_prompt_dst(self):
        mode: PromptMode = "causal_dst"
        preamble = self.simple_ft_pg.get_finetuning_preamble(mode)
        ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(self.one_turn, mode=mode)
        expected_prompt = preamble + \
                          "    # Example 1\n" + \
                          "    response = agent.handle_turn(\n" + \
                          "        belief_state=BeliefState(),\n" + \
                          "        last_system_utterance=\"\",\n" + \
                          "        user_utterance=\"I am looking for a train that leaves Cambridge going to Leicester arriving by 16:15\",\n" + \
                          "        user_intent=[agent."
        expected_completion = "find_train(arrive_by=\"16:15\", departure=\"cambridge\", destination=\"leicester\")]"
        self.assertEqual(ft_prompt, expected_prompt)
        self.assertEqual(ft_completion, expected_completion)

    def test_simple_get_dst_prompt(self):
        ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(self.one_turn,
                                                                                          mode="causal_dst")
        self.assertEqual(ft_prompt, self.simple_ft_pg.get_dst_prompt(
            turn_user_utterances=self.one_turn['user_utterances'],
            turn_system_utterances=self.one_turn['system_utterances'],
            belief_state_history=[self.one_turn['last_slot_values']],
            mode="causal_dst"
        ))

    def test_parse_act_completion_regressions(self):
        completion: str = "[Request(service='attraction', values=['area', 'type', 'entrance_fee', 'open_hours', " \
                          "'address', 'phone', 'postcode'], state={'attraction': {'name': 'la raza'}}]"
        parsed: List[Act] = self.pg.parse_sys_act_completion(completion, state={})
        print(parsed)

    def test_get_preamble(self):
        # get all values from the Literal Mode
        for mode in get_args(PromptMode):
            preamble = self.pg.get_preamble(mode)
            self.assertIsInstance(preamble, str)
            self.assertTrue(preamble)  # not blank
            exec_preamble = preamble.replace(
                "from dialogue.management",
                "from nc_latent_tod.kwargs_prompt.dialogue.management"
            )
            exec_preamble = exec_preamble.replace("from dialogue.act", "from nc_latent_tod.acts.act")
            exec("import abc\n" + exec_preamble, globals())

    def test_simple_ft_get_act_prompt(self):
        mode: PromptMode = "non_causal_sys_act_resp_only"
        ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(
            self.one_turn, mode=mode)
        self.assertEqual(ft_prompt, self.simple_ft_pg.get_sys_act_tagging_prompt(
            turn_user_utterances=self.one_turn['user_utterances'],
            turn_system_utterances=self.one_turn['system_utterances'],
            turn_system_response=self.one_turn['system_response'],
            last_turn_system_acts=self.one_turn['last_system_response_acts'],
            mode=mode,
            prior_state=self.one_turn['last_slot_values'],
            next_state=self.one_turn['slot_values']
        ))
        preamble = self.simple_ft_pg.get_finetuning_preamble(mode)
        expected_prompt = preamble + \
                          "    # Example 1\n" + \
                          "    response = agent.handle_turn(\n" + \
                          "        system_response=\"We have 77 options available. Is there a certain day you want to travel?\",\n" + \
                          "        system_acts=["
        expected_completion = "Inform(entity=Train(choice='77')), Request(service='train', values=['day'])]"
        self.assertEqual(expected_prompt, ft_prompt)
        self.assertEqual(expected_completion, ft_completion)
        mode: PromptMode = "non_causal_sys_act_resp_only_noisy_channel"
        ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(
            self.one_turn, mode=mode)
        # In noisy channel mode, our FT prompt is longer than the one we get by calling get_sys_act_tagging_prompt.
        # see test_act_noisy_channel_symmetry for a test that checks that the two are equivalent in the correct scenario
        self.assertTrue(ft_prompt.startswith(self.simple_ft_pg.get_sys_act_tagging_prompt(
            turn_user_utterances=self.one_turn['user_utterances'],
            turn_system_utterances=self.one_turn['system_utterances'],
            turn_system_response=self.one_turn['system_response'],
            last_turn_system_acts=self.one_turn['last_system_response_acts'],

            mode=mode,
            prior_state=self.one_turn['last_slot_values'],
            next_state=self.one_turn['slot_values']
        )))
        preamble = self.simple_ft_pg.get_finetuning_preamble(mode)
        expected_prompt = preamble + \
                          "    # Example 1\n" + \
                          "    response = agent.handle_turn(\n" + \
                          "        system_acts=[Inform(entity=Train(choice='77')), Request(service='train', values=['day'])],\n" + \
                          "        system_response="
        expected_completion = "\"We have 77 options available. Is there a certain day you want to travel?\""
        self.assertEqual(expected_prompt, ft_prompt)
        self.assertEqual(expected_completion, ft_completion)

    def test_act_noisy_channel_symmetry(self):
        mock_client = Mock()
        mode: PromptMode = "non_causal_sys_act_resp_only_noisy_channel"
        with_examples: List[DatasetTurn] = list(self.dataset.select(range(0, 2)))
        # looping over scenarios where we have examples and where we don't
        for examples_arg in (None, [], with_examples):
            ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(
                self.one_turn, mode=mode, examples=examples_arg)
            act_module: BatchLMClientActTagModule = BatchLMClientActTagModule(
                prompt_generator=self.simple_ft_pg,
                client=mock_client,
                examples=examples_arg,
                retriever=None,
                retrieve_k_examples=2,
                generation_cfg=GenerationConfig(
                    prompt_mode="non_causal_sys_act_resp_only",
                    noisy_channel_prompt_mode="non_causal_sys_act_resp_only_noisy_channel"
                )
            )
            act_inputs: SchemaGuidedActTaggingInputs = {
                "schema": self.schema,
                "last_system_acts": self.one_turn['last_system_response_acts'],
                "system_response": self.one_turn['system_response'],
                "user_utterances": self.one_turn['user_utterances'],
                "system_utterances": self.one_turn['system_utterances'],
                "prior_state": self.one_turn['last_slot_values'],
                "next_state": self.one_turn['slot_values'],
            }
            gold_direct_completion: str = repr(
                get_acts_from_system_acts(self.one_turn['system_response_acts'], self.schema,
                                          act_loading_context=self.act_loading_context))
            nc_preamble, nc_prompt = act_module.get_noisy_channel_prompt(task_input=act_inputs, examples=examples_arg)
            direct_comp, connective, nc_completion = act_module.get_noisy_channel_completion(task_input=act_inputs,
                                                                                             completion=gold_direct_completion)
            self.assertEqual(ft_prompt + ft_completion, nc_preamble + nc_prompt + direct_comp + connective + nc_completion)

    def test_dst_noisy_channel_symmetry(self):
        mock_client = Mock()
        mode: PromptMode = "noisy_channel_dst"
        with_examples: List[DatasetTurn] = list(self.dataset.select(range(0, 2)))
        # looping over scenarios where we have examples and where we don't
        for examples_arg in (None, [], with_examples):
            ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(
                self.one_turn, mode=mode, examples=examples_arg)
            dst_module: BatchLMClientDSTModule = BatchLMClientDSTModule(
                prompt_generator=self.simple_ft_pg,
                client=mock_client,
                examples=examples_arg,
                retriever=None,
                retrieve_k_examples=2,
                generation_cfg=GenerationConfig(
                    prompt_mode="causal_dst",
                    noisy_channel_prompt_mode="noisy_channel_dst"
                )
            )
            dst_inputs: SchemaGuidedDSTInputs = {
                "schema": self.schema,
                "user_utterances": self.one_turn['user_utterances'],
                "system_utterances": self.one_turn['system_utterances'],
                "belief_state_history": [self.one_turn['last_slot_values']],
            }
            gold_completion: str = self.simple_ft_pg.get_user_intent_str(
                self.one_turn['last_slot_values'], self.one_turn['slot_values'],
                turn_strings=self.simple_ft_pg.get_turn_strings(self.one_turn)
            )
            gold_completion = self.simple_ft_pg.get_canonical_dst_completion(
                completion=gold_completion,
                previous_state=self.one_turn['last_slot_values'],
                turn_strings=self.simple_ft_pg.get_turn_strings(self.one_turn),
                mode="causal_dst"
            )
            nc_preamble, nc_prompt = dst_module.get_noisy_channel_prompt(task_input=dst_inputs, examples=examples_arg)
            direct_comp, connective, nc_completion = dst_module.get_noisy_channel_completion(
                task_input=dst_inputs, completion=gold_completion
            )
            # The noisy channel DST method will canonicalize the completion, so we can patch the ft prompt to match
            # after checking they are the same functionally
            current_update_str: str = re.findall(r'user\_intent=\[.*\]', ft_prompt)[-1]
            ft_result = self.simple_ft_pg.parse_dst_completion(current_update_str, self.one_turn['last_slot_values'])
            gold_result = self.simple_ft_pg.parse_dst_completion(gold_completion, self.one_turn['last_slot_values'])
            self.assertDictEqual(ft_result, gold_result)
            ft_prompt = ft_prompt.replace(current_update_str, f"user_intent={gold_completion}", 1)
            self.assertEqual(ft_prompt + ft_completion, nc_preamble + nc_prompt + direct_comp + connective + nc_completion)

    def test_direct_ft_prompt_symmetries(self):
        # this test verifies that for each direct prompt mode, the FT prompt is equivalent to the task prompt
        mock_client = Mock()
        modes_to_modules: Dict[PromptMode, AbstractLMClientModule] = {
            "causal_dst": BatchLMClientDSTModule(
                prompt_generator=self.simple_ft_pg,
                client=mock_client,
                examples=None,
                retriever=None,
                retrieve_k_examples=2,
                generation_cfg=GenerationConfig(prompt_mode="causal_dst")
            ),
            "non_causal_sys_act_resp_only": BatchLMClientActTagModule(
                prompt_generator=self.simple_ft_pg,
                client=mock_client,
                examples=None,
                retriever=None,
                retrieve_k_examples=2,
                generation_cfg=GenerationConfig(prompt_mode="non_causal_sys_act_resp_only")
            ),
            "causal_sys_act_policy_from_hist": BatchLMClientPolicyModule(
                prompt_generator=self.simple_ft_pg,
                client=mock_client,
                examples=None,
                retriever=None,
                retrieve_k_examples=2,
                generation_cfg=GenerationConfig(prompt_mode="causal_sys_act_policy_from_hist")
            ),
            "response_gen_simple": BatchLMClientResponseGenModule(
                prompt_generator=self.simple_ft_pg,
                client=mock_client,
                examples=None,
                retriever=None,
                retrieve_k_examples=2,
                generation_cfg=GenerationConfig(prompt_mode="response_gen_simple")
            )
        }
        for turn in tqdm(self.dataset.select(range(12))):
            modes_to_inputs: Dict[PromptMode, GenericInputs] = {
                "causal_dst": {
                    "schema": self.schema,
                    "user_utterances": turn['user_utterances'],
                    "system_utterances": turn['system_utterances'],
                    "belief_state_history": [turn['last_slot_values']],
                },
                "non_causal_sys_act_resp_only": {
                    "schema": self.schema,
                    "last_system_acts": turn['last_system_response_acts'],
                    "system_response": turn['system_response'],
                    "user_utterances": turn['user_utterances'],
                    "system_utterances": turn['system_utterances'],
                    "prior_state": turn['last_slot_values'],
                    "next_state": turn['slot_values'],
                },
                "causal_sys_act_policy_from_hist": {
                    "schema": self.schema,
                    "last_system_acts": turn['last_system_response_acts'],
                    "user_utterances": turn['user_utterances'],
                    "system_utterances": turn['system_utterances'],
                    "prior_state": turn['last_slot_values'],
                    "next_state": turn['slot_values'],
                },
                "response_gen_simple": {
                    "schema": self.schema,
                    "last_system_acts": turn['last_system_response_acts'],
                    "user_utterances": turn['user_utterances'],
                    "system_utterances": turn['system_utterances'],
                    "prior_state": turn['last_slot_values'],
                    "next_state": turn['slot_values'],
                    "system_response_acts": turn['system_response_acts'],
                }
            }
            # second policy prompt, same inputs
            for mode in modes_to_inputs.keys():
                for examples_arg in (None, [], list(self.dataset.select(range(12, 14)))):
                    task_module = modes_to_modules[mode]
                    task_module.examples = examples_arg
                    ft_prompt, ft_completion = self.simple_ft_pg.get_finetuning_prompt_and_completion(
                        turn, mode=mode, examples=examples_arg)
                    task_preamble, task_prompt = task_module.get_task_prompt(task_input=modes_to_inputs[mode], examples=examples_arg)
                    # we prefix the task prompt with a double quote in the task, but it works out correct otherwise
                    if mode == 'response_gen_simple':
                        ft_prompt += '"'
                    self.assertEqual(ft_prompt, task_preamble + task_prompt)

    def test_parse_sys_act_regressions(self):
        self.assertListEqual(
            self.simple_ft_pg.parse_sys_act_completion("[Request(service='train', values=['departure', 'day'])"),
            [Request(service='train', values=['departure', 'day'])]
        )

    def test_dst_preamble_regression(self):
        self.maxDiff = None
        current_preamble = self.pg.get_preamble("causal_dst")
        self.assertEqual(
            current_preamble,
            DST_PREAMBLE
        )


if __name__ == '__main__':
    unittest.main()
