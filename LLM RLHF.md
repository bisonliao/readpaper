**Training language models to follow instructions with human feedback**

### Introduction

通过结合监督学习（人类演示数据）和强化学习（人类偏好排序数据）的微调方法（RLHF）, 可以使得大模型：

1. 更好的遵循人类的意图，对人类更有实质性的帮助
2. 更好的生成真实的答案
3. 减少毒性输出
4. 有很好的泛化能力，没有参与微调标注工作的其他人也认为微调后的大模型的输出更符合他们的需要

![image-20250510133101639](img/image-20250510133101639.png)

Q：RLHF是否包括step1的SFT？

A:  

- **若问流程**：RLHF完整流程包括SFT → RM训练 → RL微调。
- **若问技术核心**：RLHF特指**利用人类反馈的强化学习部分**（RM+PPO），SFT是其前置准备。

需要在具体场景中明确语境，但多数实践（如ChatGPT/InstructGPT）默认RLHF包含SFT阶段 

### Method and Experimental Details

#### method概述：

- Step 1: Collect demonstration data, and train a supervised policy. Our labelers provide demonstrations of the desired behavior on the input prompt distribution (see Section 3.2 for details on this  distribution). We then fine-tune a pretrained GPT-3 model on this data using supervised learning.

- Step 2: Collect comparison data, and train a reward model. We collect a dataset of comparisons  between model outputs, where labelers indicate which output they prefer for a given input. We then  train a reward model to predict the human-preferred output.

- Step 3: Optimize a policy against the reward model using PPO. We use the output of the
RM as a scalar reward. We fine-tune the supervised policy to optimize this reward using the PPO  algorithm (Schulman et al., 2017).

Steps 2 and 3 can be iterated continuously; more comparison data is collected on the current best  policy, which is used to train a new RM and then a new policy. In practice, most of our comparison data comes from our supervised policies, with some coming from our PPO policies.

#### 数据集

三部分数据集：

1. SFT数据集，13k条，prompt来自用户提交的API，回答来自标注人手写
2. RM（奖励模型）数据集，33k条，来自API和标注人的标注
3. PPO数据集，31k条，只需要prompt，不需要任何标注

96%是英语。

下表是来自API的prompt的归类分布：

![image-20250510154539754](img/image-20250510154539754.png)

该项目对标注者有严格的考试筛选、入职培训、全程的工作讨论。并且有专门雇佣一拨人，只负责测试，不负责标注数据，以检验微调后的模型的人类偏好是否具有泛化性。

#### 模型

##### RM训练阶段

RM模型初始化，直接使用第一步SFT后得到的模型（6B大小）去掉最后一层（词汇表分类层），改为一个标量输出层，用来输出reward值。然后进行RM训练。

RM训练是这样构造的：

![image-20250510161817037](img/image-20250510161817037.png)

##### 强化学习阶段

![image-20250510163015403](img/image-20250510163015403.png)

**bandit环境，简单的说就是回合长度为1的即时奖励的环境。**



**强化学习阶段很重要的是避免过拟合到RM：**

![image-20250510171404186](img/image-20250510171404186.png)

### Result

![image-20250510191128730](img/image-20250510191128730.png)

### Discussion

局限性：

1. 方法论，依赖一小部分标注者的倾向，不完全代表所有用户
2. 模型方面，并没有彻底做到对齐人类的偏好或者彻底的安全

### bison的实验

用一个toy problem 来体验RLHF，我的奖励模型很简单：大模型输出的文本里如果有数字，就为正奖励，否则为负奖励。鼓励大模型用数据说话。



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from datetime import datetime
import re
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 100
EPOCHS = 100
CLIP_EPS = 0.2
KL_COEFF = 0.1  # KL 散度系数
VF_COEFF = 0.5  # Value loss 系数
SAVE_PATH = "./saved_model"
GAMMA = 0.99  # Discount factor (用于 GAE 或计算 returns，当前简单 reward 未直接使用)
LR_ACTOR = 1e-5
LR_CRITIC = 1e-5

global_step = 0

# ==== Dataset ====
class MyDataset(Dataset):
    def __init__(self, tokenizer, max_prompt_len=30, num_samples=5000):
        self.tokenizer = tokenizer
        try:
            streamed_dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
            raw_dataset = [item for _, item in zip(range(num_samples), streamed_dataset)]
        except Exception as e:
            print(f"Error loading dataset: {e}. Using dummy data.")
            raw_dataset = [{"text": f"This is sample text number {i} for testing purposes."} for i in range(num_samples)]

        prompts = []
        for item in raw_dataset:
            clean_text = " ".join(item["text"].split())
            words = clean_text.split()
            if len(words) >= max_prompt_len:
                prompt = " ".join(words[:max_prompt_len])
                prompts.append(prompt)

        prompts = [p for p in prompts if p.strip()]
        if not prompts:
            raise ValueError("No valid prompts generated from the dataset.")

        encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len)
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

# ==== Reward Function ====
def compute_reward(texts):
    rewards = []
    counts = []
    for t in texts:
        matches = re.findall(r"\b\d+\b", t)
        count = len(matches)
        counts.append(count)
        if count > 0:
            reward = min(count * 0.2, 1.0)
        else:
            reward = -0.2
        rewards.append(reward)
    avg_cnt = torch.tensor(counts, dtype=torch.float32).mean()
    return torch.tensor(rewards, dtype=torch.float32).to(DEVICE), avg_cnt

# ==== Actor Model ====
class ActorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2LMHeadModel(config)

# ==== Critic Model ====
class CriticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.value_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_states = output.last_hidden_state
        if attention_mask is None:
            values = self.value_head(hidden_states[:, -1, :]).squeeze(-1)
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_token_hidden_states = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
            values = self.value_head(last_token_hidden_states).squeeze(-1)
        return values

# ==== Function to get log probabilities of a sequence ====
def get_sequence_log_probs(model, input_ids, attention_mask):
    outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    action_ids = input_ids[:, 1:].unsqueeze(-1)
    gathered_log_probs = log_probs.gather(dim=-1, index=action_ids).squeeze(-1)

    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        gathered_log_probs = gathered_log_probs * mask

    sequence_log_probs = gathered_log_probs.sum(dim=1)
    return sequence_log_probs

# ==== Training ====
def train():
    global global_step
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = GPT2Config.from_pretrained("gpt2")
    config.pad_token_id = tokenizer.eos_token_id

    actor = ActorModel(config).to(DEVICE)
    critic = CriticModel(config).to(DEVICE)
    old_actor = ActorModel(config).to(DEVICE)
    old_actor.load_state_dict(actor.state_dict())
    old_actor.eval()

    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    writer = SummaryWriter(f"logs/ppo_separate_hf_adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print("Loading dataset...")
    dataset = MyDataset(tokenizer, max_prompt_len=20, num_samples=5000)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, _ = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataset loaded.")

    for epoch in range(EPOCHS):
        print(f"Starting Epoch {epoch+1}/{EPOCHS}")
        old_actor.load_state_dict(actor.state_dict())
        old_actor.eval()

        for i, batch in enumerate(train_loader):
            global_step += 1
            prompt_ids = batch["input_ids"].to(DEVICE)
            prompt_attention_mask = batch["attention_mask"].to(DEVICE)
            prompt_len = prompt_ids.shape[1]

            # --- 1. Rollout Phase: Generate text using the *old* actor ---
            old_actor.eval()
            with torch.no_grad():
                generated_output = old_actor.transformer.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_ids = generated_output
                generated_attention_mask = (generated_ids != tokenizer.pad_token_id).long()
                full_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                response_texts = tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)
            old_actor.train()
            actor.train()

            # --- 2. Compute Rewards ---
            rewards, alignment = compute_reward(full_texts)

            # --- 3. Compute Log Probabilities and Values ---
            with torch.no_grad():
                log_probs_old = get_sequence_log_probs(old_actor, generated_ids, generated_attention_mask)
                values = critic(input_ids=generated_ids, attention_mask=generated_attention_mask)

            # --- 4. Compute Advantages ---
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- 5. PPO Optimization Phase ---
            log_probs_new = get_sequence_log_probs(actor, generated_ids, generated_attention_mask)
            values_new = critic(input_ids=generated_ids, attention_mask=generated_attention_mask)

            # --- Policy (Actor) Loss ---
            ratio = torch.exp(log_probs_new - log_probs_old.detach())
            surrogate1 = ratio * advantages.detach()
            surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages.detach()
            policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

            # --- Value (Critic) Loss ---
            value_loss = F.mse_loss(values_new, rewards)

            # --- KL Divergence Penalty ---
            kl_div = torch.mean((log_probs_old.detach() - log_probs_new))

            # --- Total Actor Loss ---
            actor_loss = policy_loss + KL_COEFF * kl_div

            # --- Backpropagation ---
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss_total = VF_COEFF * value_loss
            critic_loss_total.backward()
            optimizer_critic.step()

            # --- Logging ---
            if global_step % 20 == 0:
                print(f"Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, KL Div: {kl_div.item():.4f}, Mean Reward: {rewards.mean().item():.4f}")
                writer.add_scalar("steps/reward_mean", rewards.mean().item(), global_step)
                writer.add_scalar("steps/loss_policy", policy_loss.item(), global_step)
                writer.add_scalar("steps/loss_value", value_loss.item(), global_step)
                writer.add_scalar("steps/loss_actor_total", actor_loss.item(), global_step)
                writer.add_scalar("steps/loss_critic_total", critic_loss_total.item(), global_step)
                writer.add_scalar("steps/kl_div", kl_div.item(), global_step)
                writer.add_scalar("steps/alignment", alignment, global_step)
                writer.add_scalar("steps/advantages_mean", advantages.mean().item(), global_step)
                writer.add_text("steps/full_text", full_texts[0], global_step)

        # --- Save models after each epoch ---
        epoch_save_path_actor = os.path.join(SAVE_PATH, "actor", f"epoch_{epoch+1}")
        epoch_save_path_critic = os.path.join(SAVE_PATH, "critic", f"epoch_{epoch+1}")
        os.makedirs(epoch_save_path_actor, exist_ok=True)
        os.makedirs(epoch_save_path_critic, exist_ok=True)

        print(f"Saving models for epoch {epoch+1}...")
        actor.transformer.save_pretrained(epoch_save_path_actor)
        critic.transformer.save_pretrained(os.path.join(epoch_save_path_critic, "transformer"))
        torch.save(critic.value_head.state_dict(), os.path.join(epoch_save_path_critic, "value_head.pth"))
        tokenizer.save_pretrained(epoch_save_path_actor)
        tokenizer.save_pretrained(epoch_save_path_critic)
        print("Models saved.")

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    train()
```

