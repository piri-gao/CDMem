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
            query += f'\nKnown information about current environment:\n {known_obs_history}'
        if action_guidance_history:
            query += f'\nAction guidance about the task, notice that the number enclosed in {{}} at the end of each item with higher values indicating greater importance.:\n{action_guidance_history}.'
        query += f"\nHere is the task:\n{init_ob}"
        query += short_memories
        return query    
                
    def get_reflection_prompts(self, history_log, is_success, fewshots, local_memories):
        log_str = history_log + f'\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n'
        scenario = log_str.split("Here is the task:")[-1].strip()
        query: str = f"""
You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You need to summarize the following based on this history:

KNOWN OBS: Your understanding of the current environment. This includes two aspects: first, the location where objects are placed within the current environment; second, the functions of some items. For example, the sink basin can be used to clean lettuce, and the fridge can be used to cool a mug.
MY ACTIONS: Compact representation of actions. According to the original execution order, ignore the thought process inside and only retain the action steps. If there are adjacent actions of the same type, some simplification can be made.
REFLECTION: {"A list of key actions. A list of key actions refers to actions that you believe are essential for task completion; removing any of these actions would affect the completion of the task." 
if is_success else
'''Provide different outputs based on the type of failure: 
Planning Failure: The task planning steps have issues, such as missing steps or misunderstandings of the task. Output the current planning issues and a correct plan for the current task.
Search Failure: Continuously searching for a target but unable to find it. Output the positions already searched and the next step in the search plan.
Operation Failure: The expected feedback was not received after performing the action, such as "nothing happens." Reflect on whether the current state matches the current action. For example, attempting to take something from cabinet 1 while at the location of cabinet 4. Output this reflection.
'''

}

Here are {"two" if is_success else "three"} examples:

{fewshots}

Here is the history:

{scenario}"""

        if len(local_memories) > 0:
            
            query += '\n\For the REFLECTION, you can refer to the REFLECTION from past attempts:\n'
            for i, m in enumerate(local_memories):
                query += f'Trial #{i}: {m}\n'

        query += '\nNow write your summary with KNOWN OBS, MY ACTIONS and REFLECTION'
        return query
    
    def env_summary_prompts(self, known_obs_history, env_fewshots):
        known_obs = known_obs_history['known_obs']
        increment_known_obs = known_obs_history['increment_known_obs']
        query = f"""You have explored in various environments and gained some experiences about the functions of certain items. Please summarize the functions of these items based on your experiences.
Here are two examples:

{env_fewshots}

Here are these experiences you should summary:
        """
        for i, m in enumerate(increment_known_obs):
            query += f'Experience #{i}: {m}\n'
        if known_obs:
            query += f'At the same time, you need to refer to your past summaries of this environments: {known_obs}'
            query += '\nNow write your summary with these experiences and past summaries:'
        else:
            query += '\nNow write your summary with these experiences:'
        return query
        
    def task_summary_prompts(self, action_guidance_history, task_fewshots, is_success):
        task_type = action_guidance_history['task_type'].upper()
        action_guidance = action_guidance_history['action_guidance']
        increment_action_guidance = action_guidance_history['increment_action_guidance']
        if is_success:
            query = f"""You have performed multiple {task_type} tasks. Now, based on your previous experiences of success with this task, please summarize a concise planning guide regarding this task. An action guide should serve as a planning guide for this type of task, including what to do in the first step, the second step, and so on."""
        else:
            query = f"""You have performed multiple {task_type} tasks. Now, based on your previous experiences of failure with this task, Please summarize the precautions to avoid these failures. """
        query += f"""Each experience includes the following content:

Specific Task: The particular task content, which is a type of {task_type} task.
Your Actions: The actions you took in this experience.
Result: True for success, False for failure.
Reflection: Reflection about this experince.

Here are two examples:

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
    
        