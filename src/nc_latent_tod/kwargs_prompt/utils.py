from typing import List

from nc_latent_tod.kwargs_prompt.templates import METHOD_OR_ENTITY_CLASS_DOCSTRING_TEMPLATE
from nc_latent_tod.prompting.abstract_prompt_generator import get_example_values, clean_slot_name, get_type_hint
from nc_latent_tod.schemas.data_types import ServiceSchema, IntentSchema, SlotSchema
from nc_latent_tod.schemas.utils import get_informable_slots


def build_docstring(service_schema: ServiceSchema, intent: IntentSchema = None, informable_only: bool = False) -> str:
    """
    Build a docstring for a method or entity class from the service schema and an optional intent

    :param service_schema: The service schema
    :param intent: The intent schema within the service (optional, if docstring is for an intent method)
    :param informable_only: If True, only include informable slots in the docstring
    """
    description: str = intent['description'] if intent else service_schema['description']
    parameters: List[str] = []

    slots: List[SlotSchema]
    if informable_only:
        slots = get_informable_slots(service_schema, intents=[intent] if intent else None)
    else:
        slots = service_schema['slots']
    for slot in slots:
        type_hint: str = get_type_hint(slot, include_default_value=False)
        possible_values_str: str = get_example_values(slot, quote_values=type_hint.startswith('str'))
        parameters.append(
            f"    {clean_slot_name(slot['name'], service_schema['service_name'])}: " +
            f"{type_hint}" +
            f"\n            {slot['description']}" +
            (f"  ({possible_values_str})" if possible_values_str else "")
        )
    docstring: str = METHOD_OR_ENTITY_CLASS_DOCSTRING_TEMPLATE.format(description=description,
                                                                      parameters="\n    ".join(parameters))
    return docstring
