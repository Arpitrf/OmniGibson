import sys

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecVideoRecorder
from .grpc_client_vec_env import GRPCClientVecEnv

n_envs = int(sys.argv[1])
env = GRPCClientVecEnv("0.0.0.0:50051", n_envs)

env = VecFrameStack(env, n_stack=5)
env = VecMonitor(env)
env = VecVideoRecorder(
    env,
    f"videos",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
