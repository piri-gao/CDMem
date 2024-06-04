class HPCPromptBuilder:
    def __init__(self):
        pass
    
    def get_inference_prompts(self, init_ob, fewshots, local_memories, short_memories, known_obs_history, action_guidance_history):
        query = 'Interact with a household to solve a task. You may take maximum of 20 steps. Here are two examples.\n'
        query += fewshots
        if len(local_memories) > 0:
            query += '\n\nYour memory for the task below:'
            for i, m in enumerate(local_memories):
                query += f'\nTrial {i}:\n{m.strip()}'
        if known_obs_history:
            query += f'Known information about current environment:\n {known_obs_history}'
        if action_guidance_history:
            query += f'Action guidance about the task:\n{action_guidance_history}'
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
    
    def get_summary_prompts(self, known_obs_history, action_guidance_history, env_fewshots, task_fewshots):
        env_prompt = task_prompt = None
        if len(known_obs_history) > 0:
            env_prompt = self._env_summary_prompts(known_obs_history, env_fewshots)
        if len(action_guidance_history) > 0:
            task_prompt = self._task_summary_prompts(action_guidance_history, task_fewshots)
        return env_prompt, task_prompt
    
    def _env_summary_prompts(self, known_obs_history, env_fewshots):
        known_obs = known_obs_history['known_obs']
        increment_known_obs = known_obs_history['increment_known_obs']
        if known_obs:
            increment_known_obs.append(known_obs)
        query = f"""You have conducted multiple explorations in an environment and gathered some experiences related to it. Now, you need to summarize these experiences, forming a union of the multiple experiences, to better guide future explorations of this environment.
Here is one example:

{env_fewshots}

Here are these experiences you should summary:
        """
        for i, m in enumerate(increment_known_obs):
            query += f'Experience #{i}: {m}\n'
        query += '\nNow write your summary with these experiences:'
        return query
        
    def _task_summary_prompts(self, action_guidance_history, task_fewshots):
        task_type = action_guidance_history['task_type'].upper()
        action_guidance = action_guidance_history['action_guidance']
        increment_action_guidance = action_guidance_history['increment_action_guidance']
        query = f"""You have performed multiple {task_type} tasks. Now, based on your previous experiences with this task, please summarize a concise action guide for future actions regarding this task. This guide should include successful experiences and lessons learned from failures to help you achieve better results in future tasks.
        Each experience includes the following content:

Specific Task: The particular task content, which is a type of {task_type}.
Your Actions: The actions you took in this experience.
Result: True for success, False for failure.
Reflection: Reflection about this experince.

Here is one example:

{task_fewshots}

Here are these experiences you should summary:
        """
        for i, m in enumerate(increment_action_guidance):
            task = m['task']
            my_actions = m['my_actions']
            is_success = m['is_success']
            reflection = m['reflection']
            experience = f"""Specific Task: {task}\nYour Actions: {my_actions}\nResult: {is_success}\nReflection: {reflection}"""
            query += f'Experience #{i}\n: {experience}\n'
        if action_guidance:
            query += f'At the same time, you need to refer to your past summaries of this task: {action_guidance}'
            query += '\nRemember your summary should not exceed five points, so choose the ones you consider most important. Now write your summary with these experiences and past summaries:'
        else:
            query += '\nRemember your summary should not exceed five points, so choose the ones you consider most important. Now write your summary with these experiences:'
        return query
    
        