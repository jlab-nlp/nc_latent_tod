import itertools
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from fuzzywuzzy import process, fuzz
from num2words import num2words

# Categorical slots can plausibly be checked at parse time by a real system, where names
# would require a DB round-trip
from nc_latent_tod.ontology.abstract_ontology import AbstractDBOntology
from nc_latent_tod.resources import read_json_resource
from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.schemas.data_types import SlotSchema, ServiceSchema
from nc_latent_tod.schemas.utils import get_all_informable_slot_names

# Not sure why, but each of the booleans in the multiwoz schema also includes 'free' == 'yes'
BOOLEAN_POSSIBLE_VALUES: List[str] = ["yes", "no", "free", "dontcare"]

# service_name -> slot_name -> slot_schema
SchemaSlotIndex = Dict[str, Dict[str, SlotSchema]]

TIME_SLOTS: List[str] = ['leave at', 'arrive by', 'book time']

# two named entities forms are considered as referring to the same canonical object if adding one of the prefixes
# or suffixes maps to that entity in the DB ontology
ENTITY_NAME_PREFIXES = ['the ']
ENTITY_NAME_SUFFIXES = [" hotel", " restaurant", ' cinema', ' guest house',
                        " theatre", " airport", " street", ' gallery', ' museum', ' train station']


def insert_space(token, text):
    """
    This function was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as
    originally published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

    @article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
    }

    I believe it is also derived from the original MultiWOZ repository: https://github.com/budzianowski/multiwoz
    """
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text: str) -> str:
    """
    This function was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as
    originally published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

    @article{hu2022context,
      title={In-Context Learning for Few-Shot Dialogue State Tracking},
      author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
      journal={arXiv preprint arXiv:2203.08568},
      year={2022}
    }

    I believe it is also derived from the original MultiWOZ repository: https://github.com/budzianowski/multiwoz
    """
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)
    text = re.sub(r"guesthouse", "guest house", text)

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insert_space(token, text)

    # insert white space for 's
    text = insert_space('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    # remove the added spaces before s
    text = re.sub(' s ', 's ', text)
    text = re.sub(' s$', 's', text)

    value_replacement = {'center': 'centre',
                         'caffe uno': 'cafe uno',
                         'caffee uno': 'cafe uno',
                         'christs college': 'christ college',
                         'cambridge belfy': 'cambridge belfry',
                         'churchill college': 'churchills college',
                         'sat': 'saturday',
                         'saint johns chop shop house': 'saint johns chop house',
                         'good luck chinese food takeaway': 'good luck',
                         'asian': 'asian oriental',
                         'gallery at 12': 'gallery at 12 a high street'}

    if text in value_replacement:
        text = value_replacement[text]
    return text


def index_schema_slots(schema_json: List[ServiceSchema]) -> SchemaSlotIndex:
    """
    Parsing the contents of `schema.json` into something indexed by slot name
    :param schema_json: JSON loaded `schema.json` contents,
        see https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2/schema.json
    :return: slot-names to slot definitions
    """
    schema: SchemaSlotIndex = defaultdict(dict)
    for service in schema_json:
        for slot in service['slots']:
            schema[service['service_name'].lower()][slot['name'].lower()] = slot
    return schema


class MultiWOZOntology(AbstractDBOntology):

    known_values: Dict[str, Dict[str, Set[str]]]
    schema: SchemaSlotIndex
    min_fuzzy_match: int
    # for tracking/observation only: service -> slot -> value -> found alias
    found_matches: Optional[Dict[str, Dict[str, Dict[str, str]]]]
    valid_slots: Dict[str, Set[str]]

    def __init__(self, known_values: Dict[str, Dict[str, Set[str]]],
                 schema: List[ServiceSchema],
                 min_fuzzy_match: int = 95,
                 track_matches: bool = False):
        self.known_values = known_values
        self.min_fuzzy_match = min_fuzzy_match
        self.found_matches = defaultdict(lambda: defaultdict(dict)) if track_matches else None
        self.schema = index_schema_slots(schema)
        self.valid_slots = {k: set(v) for k, v in get_all_informable_slot_names(schema).items()}

    def is_numeric(self, service_name: str, slot_name: str) -> bool:
        if not self.is_categorical(service_name, slot_name):
            return False
        possible_values: List[str] = self.schema[service_name][slot_name]['possible_values']
        return all(value.isnumeric() for value in possible_values)

    def is_bool(self, service_name: str, slot_name: str) -> bool:
        if not self.is_categorical(service_name, slot_name):
            return False
        possible_values: List[str] = self.schema[service_name][slot_name]['possible_values']
        return all(value.lower() in BOOLEAN_POSSIBLE_VALUES for value in possible_values)

    def is_categorical(self, service_name: str, slot_name: str) -> bool:
        # service and slot are present and is_categorical is set to True
        return service_name in self.schema and slot_name in self.schema[service_name] and \
               self.schema[service_name][slot_name].get('is_categorical')

    def is_name(self, service_name: str, slot_name: str) -> bool:
        return slot_name == 'name'

    # separating for readability and testing
    @staticmethod
    def numeral_aliases(value: str) -> Set[str]:
        aliases = set()
        tokens = value.split()  # default to white-space tokenization for handling numerals
        numeric_indices: List[int] = [i for (i, token) in enumerate(tokens) if token.isnumeric()]
        # this is exhaustive, but should work generally
        for subset_size in range(len(numeric_indices) + 1):
            for combination in itertools.combinations(numeric_indices, subset_size):
                aliases.add(' '.join(num2words(token) if i in combination else token for (i, token) in
                                     enumerate(tokens)))
                # consider multi-digit tokens as having both full-number and per-digit aliases:
                # restaurant 17 = restaurant seventeen AND restaurant one seven
                aliases.add(' '.join(MultiWOZOntology._per_digit_num2words(token) if i in combination else token
                                     for (i, token) in enumerate(tokens)))
        return aliases

    @staticmethod
    def get_acceptable_aliases(value: str) -> List[str]:
        aliases = {value}
        # first, consider possible truncations of the given value (removing prefix or suffix)
        for prefix in ENTITY_NAME_PREFIXES:
            accepted_alternates = []
            if value.startswith(prefix):
                # add JUST truncating the prefix
                aliases.add(value[len(prefix):])
                # track alternates in case we need to drop prefix AND suffix
                accepted_alternates.append(value[len(prefix):])
            for suffix in ENTITY_NAME_SUFFIXES:
                if value.endswith(suffix):
                    # add JUST truncating the suffix
                    aliases.add(value[:-len(suffix)])
                    # add truncating both, if we've truncated a prefix
                    aliases.update([alt[:-len(suffix)] for alt in accepted_alternates])
        # consider all combinations of adding and removing a prefix/suffix. In a test and code we'll ensure we aren't
        # creating transformations for a single value that match 2+ distinct entities (since these should be aliases for
        # just one entity
        for alias in list(aliases):
            for prefix in ENTITY_NAME_PREFIXES:
                if not alias.startswith(prefix):
                    # prefix not present. add prefix
                    aliases.add(prefix + alias)
                    # also check if we can add suffixes WITH this prefix added
                    for suffix in ENTITY_NAME_SUFFIXES:
                        if not alias.endswith(suffix):
                            aliases.add(prefix + alias + suffix)
            # for each alias, also consider only suffixes
            for suffix in ENTITY_NAME_SUFFIXES:
                if not alias.endswith(suffix):
                    aliases.add(alias + suffix)

        # Finally, for all aliases, consider aliases for numerals to words e.g. 'restaurant 2 2' -> 'restaurant two two'
        numeral_aliases = set()
        for alias in aliases:
            numeral_aliases.update(MultiWOZOntology.numeral_aliases(alias))
        aliases.update(numeral_aliases)
        return list(aliases)

    def get_canonical(self, service_name: str, slot_name: str, value: str) -> Optional[str]:
        """
        For a given full slot name (e.g. 'hotel-name'), convert the given value into its canonical form. The canonical
        form for a slot value (e.g. name) is the form defined in the original database for entity it references. E.g:
        surface forms 'the acorn guest house', 'acorn guest house', 'the acorn guesthouse' all de-reference to
        canonical form 'acorn guest house', as defined in db/multiwoz/hotel_db.json

        :param service_name: name of the service e.g. 'hotel'
        :param slot_name: name of the slot within the domain e.g. 'name'
        :param value: the value to convert. Does not need to be a name, could be a category or timestamp
            (e.g. we handle '5:14' -> '05:14')
        :return: canonical form of the value for the given slot, or None if there is not one (which implies the value
           is not in the ontology).
        """
        if not self.is_valid_slot(service_name, slot_name):
            logging.warning(f"seeking a canonical value for an unknown service_name={service_name}, "
                            f"slot_name={slot_name}, slot_value={value}")
            return None
        if service_name in self.known_values and slot_name in self.known_values[service_name]:
            # direct match: value is already canonical
            if value in self.known_values[service_name][slot_name]:
                return value
            else:
                # Add acceptable prefixes and suffixes such that we hopefully find an exact match. A test verifies these
                # uniquely identify an object, instead of two aliases for the same value yielding different db objects
                aliases = self.get_acceptable_aliases(value)
                for alias in aliases:
                    if alias in self.known_values[service_name][slot_name]:
                        # this is the canonical alias which matches an actual DB entity name
                        if self.found_matches is not None:
                            self.found_matches[service_name][slot_name][value] = alias
                        return alias
                # No direct matches. Finally, attempt a fuzzy match (could be a mispelling, e.g. 'pizza hut fenditton'
                # vs. 'pizza hut fen ditton'
                fuzzy_matches: List[Tuple[str, str, int]] = []
                for alias in aliases:
                    # fuzz.ratio does NOT account for partial phrase matches, which is preferred, since these can
                    # have surprisingly high scores mapping from generic to specific, e.g:
                    # 'archaeology' -> 'museum of science and archaeology' is pretty high. Since we consider so many
                    # aliases, we want to be sure we are matching intended entities and not inferring from ambiguous
                    # predictions
                    best_match, best_score = process.extractOne(alias, self.known_values[service_name][slot_name],
                                                                scorer=fuzz.ratio)
                    if best_score >= self.min_fuzzy_match:
                        fuzzy_matches.append((best_match, alias, best_score))
                unique_matches: Set[str] = set(match for match, _, _ in fuzzy_matches)
                if len(unique_matches) > 1:
                    print(f"Warning: a had aliases yielding two distinct fuzzy matches. Consider increasing "
                          f"min_fuzz_value: {fuzzy_matches}")
                    # if we cannot narrow it down, say this matches neither (could maybe also pick nearest?)
                    return None
                else:
                    # all the same, just get the first
                    if fuzzy_matches:
                        match, alias, score = fuzzy_matches[0]
                        if self.found_matches is not None:
                            self.found_matches[service_name][slot_name][value] = match
                        return match
                    return None

        elif slot_name in TIME_SLOTS:
            # convert 9:00 -> 09:00
            if ':' in value and len(value) < 5:
                value = '0' + value
            # then: verify it is actually a time-stamp in (00:00 -> 23:59)
            return value if self.is_valid_time(value) or value == 'dontcare' else None
        else:
            raise ValueError(f"unexpected slot: service={service_name} slot_name={slot_name}")

    @staticmethod
    def create_ontology(min_fuzzy_match: int = 90, track_matches: bool = False) -> "MultiWOZOntology":
        known_values: Dict[str, Dict[str, Set[str]]] = defaultdict(dict)

        # read schema
        raw_schema: List[ServiceSchema] = read_multiwoz_schema()
        schema: SchemaSlotIndex = index_schema_slots(raw_schema)

        # read database files
        domain_dbs = {
            domain: read_json_resource(f"db/multiwoz/{domain}_db.json")
            for domain in ('attraction', 'bus', 'hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train')
        }

        # iterate over the slots we care about (the informable ones, for state tracking)
        time_slots = []
        location_slots: List[Tuple[str, str]] = []
        for service_name, slot_names in get_all_informable_slot_names(raw_schema).items():
            for slot_name in slot_names:
                # categorical slots define their possible values in the schema
                if schema[service_name][slot_name]['is_categorical']:
                    known_values[service_name][slot_name] = \
                        set(schema[service_name][slot_name]['possible_values'] + ['dontcare'])
                # these are time-based slots, we'll need to validate with functions vs. possible values
                elif slot_name in TIME_SLOTS:
                    time_slots.append(slot_name)
                # these are location slots, derived from located entities, fill in later.
                elif slot_name in ('departure', 'destination'):
                    location_slots.append((service_name, slot_name))
                # non-categorical slots (e.g. hotel names) do not defined possible values in schema, but we can
                # reference all values present in the database for these
                else:
                    domain_db = domain_dbs[service_name]
                    # no normalization here!
                    known_values[service_name][slot_name] = set(
                        [normalize(entity[slot_name]) for entity in domain_db] + ['dontcare']
                    )
        locations: Set[str] = {'dontcare'}
        for service_name in ('attraction', 'hospital', 'hotel', 'police', 'restaurant'):
            locations.update(known_values[service_name].get('name', []))

        # some locations exist only as referenced in departure/destination locations of trains, busses
        for domain in ('bus', 'train'):
            for journey in domain_dbs[domain]:
                locations.add(journey['destination'])
                locations.add(journey['departure'])
        for service_name, slot_name in location_slots:
            known_values[service_name][slot_name] = locations

        return MultiWOZOntology(known_values, schema=raw_schema, min_fuzzy_match=min_fuzzy_match, track_matches=track_matches)

    def is_valid_slot(self, service_name: str, slot_name: str) -> bool:
        """
        service and slot name should be specified and known to the ontology (not case sensitive)

        :param service_name: name of the service
        :param slot_name: name of the slot
        :return: whether it is a valid/known slot to the ontology
        """
        return service_name and service_name.lower() in self.valid_slots and \
               slot_name and slot_name.lower() in self.valid_slots[service_name.lower()]


if __name__ == '__main__':
    ontology: MultiWOZOntology = MultiWOZOntology.create_ontology()
    print(ontology.valid_slots)
