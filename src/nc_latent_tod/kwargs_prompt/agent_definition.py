from typing import Dict, List

from nc_latent_tod.acts.act import Act
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
        """
        user communicates no change in the dialogue state
        """
        pass

    # <PLACEHOLDER: INTENT METHODS HERE>


if __name__ == '__main__':
    agent = DialogueAgent()