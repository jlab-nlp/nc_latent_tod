ABSTRACT_METHOD_TEMPLATE: str = \
    """
    @abc.abstractmethod
    def {method_name}(self, {method_signature}) -> {return_type}:
        {docstring}
        pass
"""
# A placeholder used to add schema-specific items to the prompt, auto-derived from schema (see agent_definition.py,
# (act_definition.py, etc.)
INTENT_METHODS_PLACEHOLDER: str = "# <PLACEHOLDER: INTENT METHODS HERE>"
SERVICE_ACT_FIELDS_PLACEHOLDER: str = "# <PLACEHOLDER: SCHEMA SLOTS GO HERE>"
SERVICE_NAMES_PLACEHOLDER: str = "# <PLACEHOLDER: SERVICE NAMES GO HERE>"
SERVICE_ENTITY_PLACEHOLDER: str = "# <PLACEHOLDER: SERVICE ENTITIES GO HERE>"
EXAMPLE_ACTS_PLACEHOLDER: str = "# <PLACEHOLDER: EXAMPLE DIALOGUE ACTS GO HERE>"

# Python class and method templates, for schema-derived code generation
ALIAS_METHOD_TEMPLATE: str = \
    """
    def {method_name}(self, **kwargs):
        return self.{alias_name}(**kwargs)
"""
INFORM_CLASS_TEMPLATE: str = \
    """class {service_class_name}Inform(ServiceAct):
    def __init__(self, {slot_names_and_types}):
        super().__init__(service={service_name!r}, **locals())
"""
SERVICE_ENTITY_TEMPLATE: str = \
    """class {service_class_name}(Entity):
    {docstring}
    {parameters}
"""
METHOD_OR_ENTITY_CLASS_DOCSTRING_TEMPLATE: str = \
    """\"\"\"
        {description}
    
        Parameters:
        -----------
    {parameters}
        \"\"\""""

# parser agent isn't used in a prompt but instead in completion processing, where each of its methods (per-service,
# hence templating) is called
PARSER_AGENT_TEMPLATE: str = \
    """
@dataclass
class ParserAgent:
    state: SchemaBeliefState
    
    def no_change(self):
        pass

    {methods}
"""
PARSER_METHOD_TEMPLATE: str = \
    """
    def {intent_name}(self, {intent_args}, **kwargs):
        named_args = dict((k, v) for k, v in locals().items() if k != 'self' and k != 'kwargs')
        for key, value in named_args.items():
            if value is not None:
                if '{service_name}' not in self.state:
                       self.state['{service_name}'] = EasyDict()
                self.state['{service_name}'][key] = value
"""
