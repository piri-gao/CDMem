class HPCPromptBuilder:
    def __init__(self):
        pass
    
    def get_inference_prompts(self, init_ob, fewshots, local_memories, short_memories):
        query = 'Interact with a household to solve a task. You may take maximum of 20 steps. Here are two examples.\n'
        query += fewshots
        if len(local_memories) > 0:
            query += '\n\nYour memory for the task below:'
            for i, m in enumerate(local_memories):
                query += f'\nTrial {i}:\n{m.strip()}'
        query += f"\nHere is the task:\n{init_ob}"
        query += short_memories
        return query
                
    def get_reflection_prompts(self, log_str, fewshots, local_memories):
        scenario = log_str.split("Here is the task:")[-1].strip()
        query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You need to summarize the following based on this history:

KNOWN OBS: Describes the known perceptions of the environment. For example, what exists in the environment and where.
MY ACTIONS: Describes the actions you have taken. The descriptions of specific actions should match the history. If there is a series of similar actions that lead to the same consequences, some simplification can be made.
REFLECTION: Describes the experiences or lessons learned in this history, guiding yourself to complete the task or complete it in fewer steps.

{fewshots}


{scenario}"""

        if len(local_memories) > 0:
            
            query += '\n\For the REFLECTION, you can refer to the REFLECTION from past attempts:\n'
            for i, m in enumerate(local_memories):
                query += f'Trial #{i}: {m}\n'

        query += '\nNow write your summary with KNOWN OBS, MY ACTIONS and REFLECTION'
        return query