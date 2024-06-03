from typing import List, Dict

from nc_latent_tod.acts.act import Act, Entity
from nc_latent_tod.kwargs_prompt.dialogue.management import Agent, BeliefState, DB, Policy, Intent, update_state


# Entity definitions can be used as arguments to service dialogue acts.

# <PLACEHOLDER: SERVICE ENTITIES GO HERE>

class ServiceAct(Act):
    """ A service act communicates something about a service Entity"""
    entity: Entity = None


# ServiceAct sub-types can use informable/requestable slots as arguments when applicable.
class Inform(ServiceAct):
    """Provide information"""
    pass


class Offer(ServiceAct):
    """System provides an offer or suggestion based on results"""
    num_choices: int = None
    pass


class Confirm(ServiceAct):
    """seek confirmation of something"""
    pass


class Affirm(ServiceAct):
    """Express agreement or confirmation."""
    pass


class Negate(ServiceAct):
    """User or System denies or negates."""
    pass


class NotifySuccess(ServiceAct):
    """notify of a successful action or result."""
    pass


class NotifyFailure(ServiceAct):
    """notify of an error or failure."""
    pass

class Acknowledge(Act):
    pass


class Goodbye(Act):
    pass


class Greeting(Act):
    pass


class ThankYou(Act):
    pass


class RequestAlternatives(Act):
    """Ask for other options, alternatives, or any additional user goals"""
    pass


class Request(Act):
    """Ask for specific information or action."""
    service: str = None  # <PLACEHOLDER: SERVICE NAMES GO HERE>
    values: List[str] = None


class DialogueAgent(Agent):

    history: List[str]
    state: BeliefState
    db: DB
    policy: Policy


    def handle_turn(self, belief_state: BeliefState, last_system_utterance: str, last_system_acts: List[Act],
                    user_utterance: str, user_intent: List[Intent], db_result: List[Dict[str, str]] = None,
                    system_acts: List[Act] = None, system_response: str = None):
        self.history.append(user_utterance)
        self.state = update_state(belief_state, last_system_acts, user_intent)
        if not db_result:
            db_result = self.db.query(self.state)
        if not system_acts:
            system_acts = self.policy.next_act(self.state, db_result, history=self.history)
        if not system_response:
            system_response = self.generate_response(system_acts)
        self.history.append(system_response)
        return system_response, system_acts


if __name__ == '__main__':
    agent = DialogueAgent()

    # Some example dialogue acts:
    # <PLACEHOLDER: EXAMPLE DIALOGUE ACTS GO HERE>