import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os

# ==========================================
# 1. DATA GENERATOR (Normalized 4D Features)
# ==========================================
def get_16qam_symbols():
    coords = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    return torch.tensor([complex(r, i) for r in coords for i in coords], dtype=torch.complex64)

def generate_device_data(num_samples, h=None, beta=None, snr_db=21, is_pilot=False):
    symbols = get_16qam_symbols()
    if h is None: h = (torch.randn(1) + 1j * torch.randn(1)) / math.sqrt(2)
    if beta is None: beta = torch.empty(1).uniform_(0.05, 0.15).item()
        
    if is_pilot and num_samples > 0:
        labels = torch.arange(num_samples) % 16
    else:
        labels = torch.randint(0, 16, (num_samples,))
    
    s = symbols[labels]
    s_mag = torch.abs(s)
    x = ((4.0 * s_mag) / (1.0 + beta * (s_mag ** 2))) * torch.exp(1j * torch.angle(s))
    
    noise_var = torch.mean(torch.abs(symbols)**2) / (10 ** (snr_db / 10))
    noise = (torch.randn(num_samples) + 1j * torch.randn(num_samples)) * math.sqrt(noise_var / 2)
    
    y = h * x + noise
    
    # Safely normalized to prevent gradient explosions
    mag_norm = torch.abs(y) / 4.0 
    phase_norm = torch.angle(y) / math.pi
    
    features_4d = torch.stack((y.real, y.imag, mag_norm, phase_norm), dim=1).float()
    return features_4d, labels, h, beta

# ==========================================
# 2. ARCHITECTURES
# ==========================================

# --- BASELINE (For Joint Training & Fixed Init) ---
def init_baseline_weights():
    he = lambda out_f, in_f: (torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)).requires_grad_(True)
    return {
        'w1': he(10, 2),  'b1': torch.zeros(10, requires_grad=True),    
        'w2': he(10, 10), 'b2': torch.zeros(10, requires_grad=True),
        'w3': he(16, 10), 'b3': torch.zeros(16, requires_grad=True)
    }

def forward_baseline(x_4d, weights):
    x_2d = x_4d[:, :2] # Baseline only gets Real and Imaginary
    h1 = F.relu(F.linear(x_2d, weights['w1'], weights['b1']))
    h2 = F.relu(F.linear(h1, weights['w2'], weights['b2']))
    return F.linear(h2, weights['w3'], weights['b3'])

# --- ENHANCED NOVELTY (4D ResNet for MAML) ---
def init_enhanced_weights():
    he = lambda out_f, in_f: (torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)).requires_grad_(True)
    return {
        'w1': he(16, 4),  'b1': torch.zeros(16, requires_grad=True),    
        'w2': he(16, 16), 'b2': torch.zeros(16, requires_grad=True),
        'w3': he(16, 16), 'b3': torch.zeros(16, requires_grad=True),
        'w4': he(16, 16), 'b4': torch.zeros(16, requires_grad=True)
    }

def forward_enhanced(x_4d, weights):
    h1 = F.relu(F.linear(x_4d, weights['w1'], weights['b1']))
    h2 = F.relu(F.linear(h1, weights['w2'], weights['b2']))
    h3_pre = F.linear(h2, weights['w3'], weights['b3'])
    h3 = F.relu(h3_pre + h1) # The ResNet Skip Connection
    return F.linear(h3, weights['w4'], weights['b4'])

# ==========================================
# 3. FAST ADAPTATION & EVALUATION
# ==========================================
def fast_adapt(y_supp, l_supp, weights, inner_lr, steps, create_graph, forward_fn):
    fast_w = {k: v.clone() for k, v in weights.items()}
    if steps > 0 and len(y_supp) > 0:
        for _ in range(steps):
            logits = forward_fn(y_supp, fast_w)
            loss = F.cross_entropy(logits, l_supp)
            grads = torch.autograd.grad(loss, fast_w.values(), create_graph=create_graph)
            fast_w = {n: w - inner_lr * g for ((n, w), g) in zip(fast_w.items(), grads)}
    return fast_w

def evaluate_model(weights, forward_fn, P, payload_size=10000, lr=0.01, adapt_steps=10):
    y_pilots, labels_pilots, device_h, device_beta = generate_device_data(P, is_pilot=True)
    
    adapted_weights = fast_adapt(y_pilots, labels_pilots, weights, lr, adapt_steps, False, forward_fn)
        
    y_payload, labels_payload, _, _ = generate_device_data(payload_size, h=device_h, beta=device_beta)
    with torch.no_grad():
        preds = torch.argmax(forward_fn(y_payload, adapted_weights), dim=1)
    return torch.sum(preds != labels_payload).item() / payload_size

# ==========================================
# 4. PLOTTING THE ONLINE LEARNING CURVE
# ==========================================
def generate_online_figure(history_t, history_maml, history_joint, history_fixed):
    print("\nGenerating the Final Online Learning Curve...")
    plt.figure(figsize=(9, 6))
    
    plt.plot(history_t, history_fixed, label='Fixed Initialization (Baseline)', linestyle='--', color='green')
    plt.plot(history_t, history_joint, label='Joint Training (Baseline)', linestyle='-', marker='s', color='blue')
    plt.plot(history_t, history_maml, label='Online MAML (Proposed 4D ResNet)', linestyle='-', marker='*', markersize=10, color='darkorange')
    
    plt.yscale('log') 
    plt.xlabel('Time Slots (New Device Connections)', fontsize=12)
    plt.ylabel('Probability of symbol error', fontsize=12)
    plt.title('Online Base Station Adaptation (P=16 Pilots)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/Figure_Novelty_Online.png', dpi=300)
    plt.show()

# ==========================================
# 5. THE ONLINE TRAINING LOOP
# ==========================================
def run_online_realistic_scenario():
    time_slots = 3000      
    batch_size = 20        
    N_tr = 16 
    N_te = 16
    inner_lr = 0.01
    meta_lr = 0.001
    
    # 4D ResNet needs exactly 3 steps to adapt perfectly
    sync_steps = 3 
    
    maml_weights = init_enhanced_weights()
    joint_weights = init_baseline_weights()
    fixed_weights = init_baseline_weights()
    
    maml_opt = torch.optim.Adam(maml_weights.values(), lr=meta_lr) 
    joint_opt = torch.optim.Adam(joint_weights.values(), lr=meta_lr)
    
    print("Starting Continuous Online Streaming...")
    print("Simulating 3000 sequential devices connecting to the Base Station.")
    
    replay_buffer = [] 
    history_t, history_maml, history_joint, history_fixed = [], [], [], []
    num_eval_devices = 20

    for t in range(time_slots):
        # 1. A new device connects
        _, _, h_new, beta_new = generate_device_data(1)
        replay_buffer.append((h_new, beta_new))
        
        # 2. Update the brains using a historical batch
        batch = random.sample(replay_buffer, min(batch_size, len(replay_buffer)))
        
        maml_loss = 0.0; joint_loss = 0.0
        for h_k, beta_k in batch:
            # Joint Training (Baseline 2D)
            y_j, labels_j, _, _ = generate_device_data(N_tr + N_te, h=h_k, beta=beta_k, is_pilot=True)
            joint_loss += F.cross_entropy(forward_baseline(y_j, joint_weights), labels_j)
            
            # Proposed MAML (4D ResNet)
            y_tr, labels_tr, _, _ = generate_device_data(N_tr, h=h_k, beta=beta_k, is_pilot=True)
            y_te, labels_te, _, _ = generate_device_data(N_te, h=h_k, beta=beta_k, is_pilot=True)
            
            fast_w = fast_adapt(y_tr, labels_tr, maml_weights, inner_lr, sync_steps, True, forward_enhanced)
            maml_loss += F.cross_entropy(forward_enhanced(y_te, fast_w), labels_te)
            
        joint_opt.zero_grad(); (joint_loss / len(batch)).backward(); joint_opt.step()
        maml_opt.zero_grad(); (maml_loss / len(batch)).backward(); maml_opt.step()
        
        # 3. Periodic Evaluation (Every 100 slots)
        if (t + 1) % 100 == 0:
            # Novelty: Evaluated at 3 steps
            avg_maml = sum(evaluate_model(maml_weights, forward_enhanced, 16, adapt_steps=sync_steps) for _ in range(num_eval_devices)) / num_eval_devices
            
            # Baselines: Evaluated at 500 steps (they have no meta-initialization)
            avg_joint = sum(evaluate_model(joint_weights, forward_baseline, 16, adapt_steps=500) for _ in range(num_eval_devices)) / num_eval_devices
            avg_fixed = sum(evaluate_model(fixed_weights, forward_baseline, 16, adapt_steps=500) for _ in range(num_eval_devices)) / num_eval_devices
            
            history_t.append(t + 1)
            history_maml.append(avg_maml)
            history_joint.append(avg_joint)
            history_fixed.append(avg_fixed)
            
            print(f"Time Slot {t+1:4d} | Fixed SER: {avg_fixed:.4f} | Joint SER: {avg_joint:.4f} | Enhanced MAML SER: {avg_maml:.4f}")

    print("\nOnline Streaming Complete!")
    generate_online_figure(history_t, history_maml, history_joint, history_fixed)

if __name__ == "__main__":
    run_online_realistic_scenario()