import abc
from typing import List, Callable, Any

from nc_latent_tod.acts.act import Act
from nc_latent_tod.data_types import SchemaBeliefState, ValidActsRepresentation


class AbstractNormalizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def normalize(self, raw_parse: SchemaBeliefState) -> SchemaBeliefState:
        """
        Normalizer addresses issues like typos in a candidate parse. The general pipeline goes like:

        1. given: completion (string) from a model
        2. raw_parse = parse(completion) -> MultiWOZDict (initial parse based on completion string)
        3. normalized_parse = normalize(raw_parse) -> MultiWOZDict (a parse that is ready for system use/auto eval)

        This is an interface for defining different approaches to step 3

        :param raw_parse: MultiWOZDict containing potentially un-normalized slot values
        :return: normalized dictionary ready for system use/eval
        """
        pass

    @abc.abstractmethod
    def normalize_acts(self, acts: ValidActsRepresentation,
                       telemetry_hook_for_removed_slot_pairs: Callable[[str, Any], Any] = None) -> List[Act]:
        """
        Normalizer addresses issues like typos in a candidate parse. The general pipeline goes like:

        1. given: completion (string) from a model
        2. raw_parse = parse(completion) -> Acts (initial parse based on completion string)
        3. normalized_parse = normalize(raw_parse) -> Acts (a parse that is ready for system use/auto eval)

        This is an interface for defining different approaches to step 3

        :param acts: Acts containing potentially un-normalized act types, slots or values
        :param telemetry_hook_for_removed_slot_pairs: an optional function to call when a slot pair is removed, i.e.
            for logging or other processing. The function should take two arguments: the slot name and the slot value.
            Note: this does NOT currently distinguish between act slots and entity slots: e.g:
               - Confirm(entity=Hotel(bad_slot='bad_value')) would call the hook with ('bad_slot', 'bad_value')
               - Confirm(bad_slot='bad_value') would ALSO call the hook with ('bad_slot', 'bad_value')
        :return: normalized Acts ready for system use/eval
        """
        pass
