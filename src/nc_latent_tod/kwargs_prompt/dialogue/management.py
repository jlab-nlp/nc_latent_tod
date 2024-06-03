from typing import Any

BeliefState = Any
update_state = Any
DB = Any
Policy = Any
Intent = Any
DBResult = Any

class Agent:
    def generate_response(self, system_acts) -> str:
        return ""