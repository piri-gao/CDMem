class ReflectPromptBuilder:
    def __init__(self):
        pass
    
    def get_inference_prompts(self, init_ob, fewshots, local_memories, short_memories, task_description, action_guides):
        query = f"""
        You are the agent to interact in a household to solve a task. Here is an example. Please read this example carefully and learn valid actions and the environment.

        {fewshots}

        Goal Task: Now, based on the task background, task instruction, reference exemplars, functions of containers, locations of items and past reflections to output the correct action.
        """
        if len(local_memories) > 0:
            query += '\n\nYour memory for the task below:'
            for i, m in enumerate(local_memories):
                query += f'\nTrial {i}:\n{m.strip()}'
        query += f"Here is action guide:\n{action_guides}\n"
        query += f"\nHere is your real task:\n{init_ob}\n{task_description}\n"
        query += short_memories
        return query
                
    def get_reflection_prompts(self, log_str, fewshots, local_memories):
        scenario = log_str.split("Here is the task:")[-1].strip()
        query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{fewshots}

{scenario}"""

        if len(local_memories) > 0:
            query += '\n\nPlans from past attempts:\n'
            for i, m in enumerate(local_memories):
                query += f'Trial #{i}: {m}\n'

        query += '\n\nNew plan:'
        return query

        