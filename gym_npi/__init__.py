from gym.envs.registration import register

register(
    id='npi-v0',
    entry_point='gym_npi.envs:NPIEnv',
)
register(
    id='npi-add-v0',
    entry_point='gym_npi.envs:NPIAddEnv',
)