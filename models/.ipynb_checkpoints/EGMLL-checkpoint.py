# ############################################model log logistics###############################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def logsumexp(a, dim, b):
#     a_max = torch.max(a, dim=dim, keepdims=True)[0]
#     out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-12)
#     out = out + a_max
#     return out

# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout=0.25):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hidden_size, output_size)
#         )
#     def forward(self, x): return self.net(x)

# class EGMLL(nn.Module):

#     def __init__(self, backbone,
#                  input_size=384, hidden_size=256,
#                  E=2, K=50, param_share=('alpha','beta'),
#                  dropout=0.1, **_ignored):
#         super().__init__()
#         self.E, self.K = E, K
#         self.param_share = tuple(param_share) if isinstance(param_share, (list, tuple)) else (param_share,)
#         self.backbone = backbone

#         # gating network
#         self.gate = MLP(input_size, hidden_size, E, dropout)

#         # per-expert heads: w always present, plus non-shared alpha/beta if requested
#         non_shared_count = sum(1 for p in ['alpha', 'beta'] if p not in self.param_share)
#         head_dim = self.K * (1 + non_shared_count)
#         self.heads = nn.ModuleList([MLP(input_size, hidden_size, head_dim, dropout) for _ in range(E)])

#         # optionally shared alpha / beta (learnable global)
#         shared = {}
#         if 'alpha' in self.param_share:
#             alpha_init = torch.linspace(0.5, 3.0, K)
#             shared['alpha'] = nn.Parameter(torch.log(F.softplus(alpha_init).unsqueeze(0) + 1e-6))
#         if 'beta' in self.param_share:
#             beta_init = torch.linspace(5.0, 50.0, K)
#             shared['beta'] = nn.Parameter(torch.log(beta_init.unsqueeze(0) + 1e-6))
#         self.shared = nn.ParameterDict(shared)

#         # allow small linear transforms of shared params
#         self.layer_alpha = nn.Linear(K, K, bias=False)
#         self.layer_beta = nn.Linear(K, K, bias=False)

#         # debugging counters
#         self._debug_printed = False
#         self._debug_count = 0

#     # --- PDF / CDF utilities ---
#     def _get_pdf_cdf(self, alpha, beta, t):
#         """
#         Compute log-PDF and CDF of the log-logistic component(s).
#         alpha: (B,K)
#         beta : (B,K)
#         t    : (B,) or (B,T)
#         returns:
#             if t (B,) -> log_pdf (B,K), cdf (B,K)
#             if t (B,T) -> log_pdf (B,T,K), cdf (B,T,K)
#         """
#         # Replace NaNs if present
#         if torch.any(torch.isnan(alpha)) or torch.any(torch.isnan(beta)):
#             alpha = torch.nan_to_num(alpha, nan=2.0)
#             beta = torch.nan_to_num(beta, nan=1.0)

#         # Transform to positive domain but with relaxed lower bounds to avoid collapse
#         # (you can reduce upper bounds later if necessary)
#         alpha_pos = F.softplus(alpha).clamp(min=1e-3, max=50.0)  
#         beta_pos  = F.softplus(beta).clamp(min=1e-3, max=1e6)    

#         if t.dim() == 1:
#             # scalar-per-sample case: t (B,)
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)  
#             # compute log_ratio = log(t/beta) with safe logs
#             log_ratio = (torch.log(t_pos) - torch.log(beta_pos)).clamp(-700, 700)  
#             z = (alpha_pos * log_ratio).clamp(-700, 700)  
#             cdf = torch.sigmoid(z)  
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)) -
#                        torch.log(beta_pos.clamp(min=1e-12)) +
#                        (alpha_pos - 1.0) * log_ratio -
#                        log_denom)  
#             return log_pdf, cdf

#         elif t.dim() == 2:
#             # vector-of-times case: t (B,T)
#             t_pos = t.clamp(min=1e-6).unsqueeze(-1)     
#             beta_exp = beta_pos.unsqueeze(1)            
#             # sanity check: batch dim must match
#             if t_pos.shape[0] != beta_exp.shape[0]:
#                 raise RuntimeError(f"_get_pdf_cdf batch mismatch: t_pos {t_pos.shape} vs beta {beta_exp.shape}")
#             log_ratio = (torch.log(t_pos) - torch.log(beta_exp)).clamp(-700, 700)  
#             z = (alpha_pos.unsqueeze(1) * log_ratio).clamp(-700, 700)              
#             cdf = torch.sigmoid(z)  
#             log_denom = 2.0 * torch.log1p(torch.exp(z))
#             log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)).unsqueeze(1) -
#                        torch.log(beta_pos.clamp(min=1e-12)).unsqueeze(1) +
#                        (alpha_pos.unsqueeze(1) - 1.0) * log_ratio -
#                        log_denom)  
#             return log_pdf, cdf
#         else:
#             raise ValueError(f"_get_pdf_cdf: unexpected t.dim()={t.dim()}")

#     def _cdf_single(self, w, alpha, beta, t):
#         """
#         Weighted CDF for a single MDN (after marginalizing K components).
#         w: (B,K)
#         alpha, beta: (B,K)
#         t: (B,) or (B,T)
#         returns: (B,) or (B,T)
#         """
#         if t.dim() == 1:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)  
#             if w.shape != cdf.shape:
#                 print(f"DEBUG _cdf_single(scalar) shape mismatch: w {w.shape} vs cdf {cdf.shape}")
#             return (w * cdf).sum(-1)  
#         elif t.dim() == 2:
#             _, cdf = self._get_pdf_cdf(alpha, beta, t)  
#             w_exp = w.unsqueeze(1)  
#             if not (w_exp.shape[0] == cdf.shape[0] and w_exp.shape[2] == cdf.shape[2]):
#                 raise RuntimeError(f"_cdf_single broadcast mismatch: w_exp {w_exp.shape} vs cdf {cdf.shape}")
#             return (w_exp * cdf).sum(-1)  
#         else:
#             raise ValueError(f"_cdf_single: unexpected t.dim()={t.dim()}")

#     def cdf(self, params, t):
#         return self._cdf_single(params['w'], params['alpha'], params['beta'], t)

#     def log_prob(self, params, t):
#         log_pdf, _ = self._get_pdf_cdf(params['alpha'], params['beta'], t)
#         w = params['w']
#         if t.dim() == 1:
#             return logsumexp(log_pdf, dim=-1, b=w).squeeze(-1).clamp(min=-1e3, max=1e3)
#         else:
#             b = w.unsqueeze(1)  
#             return logsumexp(log_pdf, dim=-1, b=b).squeeze(-1).clamp(min=-1e3, max=1e3)

#     def forward(self, x, **kw):
#         """
#         x -> backbone -> features h (B, D)
#         returns params dict with 'w', 'alpha', 'beta', optional subloss and extra_losses
#         """
#         h_back = self.backbone(x, **kw)
#         if isinstance(h_back, dict):
#             subloss = h_back.get('loss')
#             h = h_back['feat']
#         else:
#             subloss, h = None, h_back

#         B = h.size(0)
#         if B == 0 or not torch.isfinite(h).all():
#             device = h.device
#             safe = {
#                 'w': torch.ones(B, self.K, device=device) / self.K,
#                 'alpha': torch.ones(B, self.K, device=device) * 2.0,
#                 'beta': torch.ones(B, self.K, device=device) * 10.0
#             }
#             return safe, subloss, {'L_ent': torch.tensor(0.0, device=device)}

#         G = self.gate(h).softmax(-1) 

#         expert_params = []
#         for e in range(self.E):
#             out = self.heads[e](h)
#             if not torch.isfinite(out).all():
#                 print(f"Warning: Non-finite out for expert {e}; zeroing.")
#                 out = torch.zeros_like(out)

#             offset = 0
#             p = {}
#             # mixture weights w
#             p['w'] = F.softmax(out[:, offset:offset + self.K], dim=-1)
#             offset += self.K

#             # alpha / beta either shared or coming from head output
#             for name in ['alpha', 'beta']:
#                 if name in self.param_share:
#                     p[name] = self.shared[name].expand(B, -1)
#                 else:
#                     p[name] = out[:, offset:offset + self.K]
#                     offset += self.K

#             if 'alpha' in self.param_share:
#                 p['alpha'] = self.layer_alpha(p['alpha'])
#             if 'beta' in self.param_share:
#                 p['beta'] = self.layer_beta(p['beta'])

#             if torch.any(torch.isnan(p.get('alpha', torch.zeros(1)))) or torch.any(torch.isnan(p.get('beta', torch.zeros(1)))):
#                 p['alpha'] = torch.zeros(B, self.K, device=h.device)
#                 p['beta'] = torch.zeros(B, self.K, device=h.device)

#             # DEBUG: print raw stats occasionally (before softplus/clamp) to diagnose collapse
#             if self._debug_count % 200 == 0:
#                 with torch.no_grad():
#                     raw_a = p['alpha'].mean().item() if p['alpha'].numel() > 1 else float(p['alpha'].item())
#                     raw_a_std = p['alpha'].std().item() if p['alpha'].numel() > 1 else 0.0
#                     raw_b = p['beta'].mean().item() if p['beta'].numel() > 1 else float(p['beta'].item())
#                     raw_b_std = p['beta'].std().item() if p['beta'].numel() > 1 else 0.0
#                     print(f"DEBUG raw params expert {e}: alpha mean {raw_a:.4f} std {raw_a_std:.4f}; beta mean {raw_b:.4f} std {raw_b_std:.4f}")

#             # apply softplus + clamp (relaxed lower bounds)
#             p['alpha'] = F.softplus(p['alpha']).clamp(0.1, 10.0)  
#             p['beta']  = F.softplus(p['beta']).clamp(1.0, 200.0)

#             expert_params.append(p)

#         if len(expert_params) == 0:
#             params = {
#                 'w': torch.ones(B, self.K, device=h.device) / self.K,
#                 'alpha': torch.ones(B, self.K, device=h.device) * 2.0,
#                 'beta': torch.ones(B, self.K, device=h.device) * 10.0
#             }
#             extra_losses = {'L_ent': torch.tensor(0.0, device=h.device)}
#             return params, subloss, extra_losses

#         # Stack per-expert params: (B,E,K)
#         w_stack = torch.stack([p['w'] for p in expert_params], dim=1)
#         alpha_stack = torch.stack([p['alpha'] for p in expert_params], dim=1)
#         beta_stack = torch.stack([p['beta'] for p in expert_params], dim=1)

#         # apply gating to aggregate patient-specific parameters
#         G_expanded = G.unsqueeze(-1)
#         w_weighted = torch.sum(G_expanded * w_stack, dim=1)
#         params_w = F.softmax(w_weighted, dim=-1)

#         alpha_weighted = torch.sum(G_expanded * alpha_stack, dim=1)  
#         beta_weighted  = torch.sum(G_expanded * beta_stack, dim=1)   

#         # optional small noise during training to encourage diversity
#         if self.training:
#             noise_scale = 0.02
#             alpha_weighted = alpha_weighted + torch.randn_like(alpha_weighted) * noise_scale
#             beta_weighted  = beta_weighted  + torch.randn_like(beta_weighted)  * (noise_scale * 5.0)

#         params = {
#             'w': params_w,
#             'alpha': F.softplus(alpha_weighted).clamp(0.1, 10.0),
#             'beta' : F.softplus(beta_weighted).clamp(1.0, 200.0)
#         }

#         # entropy loss on gating (encourage soft gating)
#         L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()
#         extra_losses = {'L_ent': L_ent * 0.001}

#         self._debug_count += 1
#         if self._debug_count % 50 == 0:
#             print(f"DEBUG Forward #{self._debug_count} - alpha std {params['alpha'].std():.4f}, beta std {params['beta'].std():.4f}, gate std {G.std():.4f}")
#         return params, subloss, extra_losses

#     def train_step(self, x_dict, t, c, constant_dict):
#         params, subloss, extras = self(**x_dict)
#         # NaN recovery
#         if torch.any(torch.isnan(params['alpha'])) or torch.any(torch.isnan(params['beta'])) or torch.any(torch.isnan(params['w'])):
#             params['alpha'] = torch.ones_like(params['alpha']) * 2.0
#             params['beta']  = torch.ones_like(params['beta']) * 10.0
#             params['w']     = torch.ones_like(params['w']) / self.K

#         surv = 1.0 - self.cdf(params, t)
#         pdf_log = self.log_prob(params, t)
#         pdf = torch.exp(pdf_log)

#         pdf = pdf.clamp(min=1e-8, max=1e6)
#         surv = surv.clamp(min=1e-8, max=1.0)

#         return {
#             't': t,
#             'c': c,
#             'pdf': pdf,
#             'survival_func': surv,
#             'subloss': subloss,
#             'L_ent': extras['L_ent']
#         }

#     def eval_step(self, x_dict, t, c, constant_dict):
#         """
#         Evaluate and return:
#          - cum_hazard_seqs: (T, B)
#          - survival_seqs: (T, B)
#          - eval_t: canonical 1-D tensor (T,)
#         """
#         with torch.no_grad():
#             params, _, extras = self(**x_dict)
#             B = t.size(0)
#             device = t.device
#             outputs = {"t": t, "c": c, "L_ent": extras['L_ent']}

#             raw_eval_t = constant_dict["eval_t"].to(device)

#             # Robust shaping for eval_t -> produce eval_t shaped (B, T)
#             if raw_eval_t.dim() == 2 and raw_eval_t.size(0) == B:
#                 eval_t = raw_eval_t
#             else:
#                 t_vec = raw_eval_t.reshape(-1)
#                 eval_t = t_vec.unsqueeze(0).expand(B, t_vec.numel()).contiguous()

#             # safety clamp
#             if eval_t.min() < 1e-6:
#                 eval_t = eval_t.clamp(min=1e-6)

#             # median time per patient -> risk score
#             median_t = torch.median(eval_t, dim=-1, keepdim=True)[0]  
#             cdf_median = self.cdf(params, median_t)  
#             surv_median = 1.0 - cdf_median
#             risk_scores = -torch.log(surv_median.clamp(min=1e-12))

#             # full hazard grid
#             cdf = self.cdf(params, eval_t)  
#             surv = 1.0 - cdf
#             haz = -torch.log(surv.clamp(min=1e-12))  

#             # Put patient-specific risk in FIRST column (CIndexMeter expectation)
#             haz_fixed = haz.clone()
#             haz_fixed[:, 0] = risk_scores.squeeze(-1)
#             outputs["cum_hazard_seqs"] = haz_fixed.transpose(0, 1).contiguous()  

#             # survival sequences on uniform grid
#             t_min = constant_dict["t_min"].to(device)
#             t_max = constant_dict["t_max"].to(device)
#             num_int_steps = int(constant_dict["NUM_INT_STEPS"])
#             grid = torch.linspace(float(t_min), float(t_max), num_int_steps, device=device)
#             surv_grid = 1.0 - self.cdf(params, grid.unsqueeze(0).repeat(B, 1))
#             outputs["survival_seqs"] = surv_grid.transpose(0, 1).contiguous()  

#             # optional eps-limited grids
#             for eps in (0.1, 0.2, 0.3, 0.4, 0.5):
#                 key = f"t_max_{eps}"
#                 if key in constant_dict:
#                     t_max_eps = constant_dict[key].to(device)
#                     grid_eps = torch.linspace(float(t_min), float(t_max_eps), num_int_steps, device=device)
#                     surv_eps = 1.0 - self.cdf(params, grid_eps.unsqueeze(0).repeat(B, 1))
#                     outputs[f"survival_seqs_{eps}"] = surv_eps.transpose(0, 1).contiguous()

#             # IMPORTANT: emit canonical 1-D eval_t so metrics get exact grid (not replicated (B,T))
#             # Use the first row as canonical grid
#             outputs["eval_t"] = eval_t[0].detach().cpu()

#             if not self._debug_printed:
#                 self._debug_printed = True
#                 print(f"DEBUG eval_step: median_t={median_t[0,0]:.4f}, risk min {risk_scores.min():.4f}, max {risk_scores.max():.4f}")

#             return outputs

#     def predict_step(self, x_dict):
#         device = x_dict['x'].device
#         with torch.no_grad():
#             params, _, _ = self(**x_dict)
#             t = torch.arange(0.1, 220.1, 0.1, device=device).unsqueeze(0)
#             surv = 1.0 - self.cdf(params, t)
#         return {'t': t.squeeze(0), 'p_survival': surv.squeeze(0)}

############################################model log logistics###############################
import torch
import torch.nn as nn
import torch.nn.functional as F


def logsumexp(a, dim, b):
    a_max = torch.max(a, dim=dim, keepdims=True)[0]
    out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-12)
    out = out + a_max
    return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class EGMLL(nn.Module):
    def __init__(self, backbone,
                 input_size=384, hidden_size=256,
                 E=2, K=50,
                 dropout=0.1, **_ignored):
        """
        No parameter sharing for alpha and beta.
        Each expert has its own alpha, beta heads.
        """
        super().__init__()
        self.E, self.K = E, K
        self.backbone = backbone

        # Gating network
        self.gate = MLP(input_size, hidden_size, E, dropout)

        # Each expert outputs (w, alpha, beta)
        head_dim = K * 3  # weights + alpha + beta
        self.heads = nn.ModuleList([MLP(input_size, hidden_size, head_dim, dropout)
                                    for _ in range(E)])

        # Debug
        self._debug_printed = False
        self._debug_count = 0

    # --- PDF / CDF utilities ---
    def _get_pdf_cdf(self, alpha, beta, t):
        if torch.any(torch.isnan(alpha)) or torch.any(torch.isnan(beta)):
            alpha = torch.nan_to_num(alpha, nan=2.0)
            beta = torch.nan_to_num(beta, nan=1.0)

        alpha_pos = F.softplus(alpha).clamp(min=1e-3, max=50.0)
        beta_pos = F.softplus(beta).clamp(min=1e-3, max=1e6)

        if t.dim() == 1:
            t_pos = t.clamp(min=1e-6).unsqueeze(-1)
            log_ratio = (torch.log(t_pos) - torch.log(beta_pos)).clamp(-700, 700)
            z = (alpha_pos * log_ratio).clamp(-700, 700)
            cdf = torch.sigmoid(z)
            log_denom = 2.0 * torch.log1p(torch.exp(z))
            log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)) -
                       torch.log(beta_pos.clamp(min=1e-12)) +
                       (alpha_pos - 1.0) * log_ratio -
                       log_denom)
            return log_pdf, cdf

        elif t.dim() == 2:
            t_pos = t.clamp(min=1e-6).unsqueeze(-1)
            beta_exp = beta_pos.unsqueeze(1)
            if t_pos.shape[0] != beta_exp.shape[0]:
                raise RuntimeError(f"_get_pdf_cdf batch mismatch: t_pos {t_pos.shape} vs beta {beta_exp.shape}")
            log_ratio = (torch.log(t_pos) - torch.log(beta_exp)).clamp(-700, 700)
            z = (alpha_pos.unsqueeze(1) * log_ratio).clamp(-700, 700)
            cdf = torch.sigmoid(z)
            log_denom = 2.0 * torch.log1p(torch.exp(z))
            log_pdf = (torch.log(alpha_pos.clamp(min=1e-12)).unsqueeze(1) -
                       torch.log(beta_pos.clamp(min=1e-12)).unsqueeze(1) +
                       (alpha_pos.unsqueeze(1) - 1.0) * log_ratio -
                       log_denom)
            return log_pdf, cdf
        else:
            raise ValueError(f"_get_pdf_cdf: unexpected t.dim()={t.dim()}")

    def _cdf_single(self, w, alpha, beta, t):
        if t.dim() == 1:
            _, cdf = self._get_pdf_cdf(alpha, beta, t)
            return (w * cdf).sum(-1)
        elif t.dim() == 2:
            _, cdf = self._get_pdf_cdf(alpha, beta, t)
            w_exp = w.unsqueeze(1)
            return (w_exp * cdf).sum(-1)
        else:
            raise ValueError(f"_cdf_single: unexpected t.dim()={t.dim()}")

    def cdf(self, params, t):
        return self._cdf_single(params['w'], params['alpha'], params['beta'], t)

    def log_prob(self, params, t):
        log_pdf, _ = self._get_pdf_cdf(params['alpha'], params['beta'], t)
        w = params['w']
        if t.dim() == 1:
            return logsumexp(log_pdf, dim=-1, b=w).squeeze(-1).clamp(min=-1e3, max=1e3)
        else:
            b = w.unsqueeze(1)
            return logsumexp(log_pdf, dim=-1, b=b).squeeze(-1).clamp(min=-1e3, max=1e3)

    def forward(self, x, **kw):
        h_back = self.backbone(x, **kw)
        if isinstance(h_back, dict):
            subloss = h_back.get('loss')
            h = h_back['feat']
        else:
            subloss, h = None, h_back

        B = h.size(0)
        if B == 0 or not torch.isfinite(h).all():
            device = h.device
            safe = {
                'w': torch.ones(B, self.K, device=device) / self.K,
                'alpha': torch.ones(B, self.K, device=device) * 2.0,
                'beta': torch.ones(B, self.K, device=device) * 10.0
            }
            return safe, subloss, {'L_ent': torch.tensor(0.0, device=device)}

        G = self.gate(h).softmax(-1)

        expert_params = []
        for e in range(self.E):
            out = self.heads[e](h)
            if not torch.isfinite(out).all():
                out = torch.zeros_like(out)

            offset = 0
            p = {}
            p['w'] = F.softmax(out[:, offset:offset + self.K], dim=-1)
            offset += self.K
            p['alpha'] = out[:, offset:offset + self.K]
            offset += self.K
            p['beta'] = out[:, offset:offset + self.K]

            # Clamp via softplus
            p['alpha'] = F.softplus(p['alpha']).clamp(0.1, 10.0)
            p['beta'] = F.softplus(p['beta']).clamp(1.0, 200.0)

            expert_params.append(p)

        w_stack = torch.stack([p['w'] for p in expert_params], dim=1)
        alpha_stack = torch.stack([p['alpha'] for p in expert_params], dim=1)
        beta_stack = torch.stack([p['beta'] for p in expert_params], dim=1)

        G_expanded = G.unsqueeze(-1)
        w_weighted = torch.sum(G_expanded * w_stack, dim=1)
        params_w = F.softmax(w_weighted, dim=-1)
        alpha_weighted = torch.sum(G_expanded * alpha_stack, dim=1)
        beta_weighted = torch.sum(G_expanded * beta_stack, dim=1)

        if self.training:
            noise_scale = 0.02
            alpha_weighted = alpha_weighted + torch.randn_like(alpha_weighted) * noise_scale
            beta_weighted = beta_weighted + torch.randn_like(beta_weighted) * (noise_scale * 5.0)

        params = {
            'w': params_w,
            'alpha': F.softplus(alpha_weighted).clamp(0.1, 10.0),
            'beta': F.softplus(beta_weighted).clamp(1.0, 200.0)
        }

        L_ent = -(G * (G + 1e-8).log()).sum(-1).mean()
        extra_losses = {'L_ent': L_ent * 0.001}

        self._debug_count += 1
        if self._debug_count % 50 == 0:
            print(f"DEBUG Forward #{self._debug_count} - alpha std {params['alpha'].std():.4f}, beta std {params['beta'].std():.4f}, gate std {G.std():.4f}")
        return params, subloss, extra_losses

    def train_step(self, x_dict, t, c, constant_dict):
        params, subloss, extras = self(**x_dict)
        if torch.any(torch.isnan(params['alpha'])) or torch.any(torch.isnan(params['beta'])) or torch.any(torch.isnan(params['w'])):
            params['alpha'] = torch.ones_like(params['alpha']) * 2.0
            params['beta'] = torch.ones_like(params['beta']) * 10.0
            params['w'] = torch.ones_like(params['w']) / self.K

        surv = 1.0 - self.cdf(params, t)
        pdf_log = self.log_prob(params, t)
        pdf = torch.exp(pdf_log).clamp(min=1e-8, max=1e6)
        surv = surv.clamp(min=1e-8, max=1.0)

        return {
            't': t,
            'c': c,
            'pdf': pdf,
            'survival_func': surv,
            'subloss': subloss,
            'L_ent': extras['L_ent']
        }

    def eval_step(self, x_dict, t, c, constant_dict):
        with torch.no_grad():
            params, _, extras = self(**x_dict)
            B = t.size(0)
            device = t.device
            outputs = {"t": t, "c": c, "L_ent": extras['L_ent']}

            raw_eval_t = constant_dict["eval_t"].to(device)
            if raw_eval_t.dim() == 2 and raw_eval_t.size(0) == B:
                eval_t = raw_eval_t
            else:
                t_vec = raw_eval_t.reshape(-1)
                eval_t = t_vec.unsqueeze(0).expand(B, t_vec.numel()).contiguous()

            eval_t = eval_t.clamp(min=1e-6)

            median_t = torch.median(eval_t, dim=-1, keepdim=True)[0]
            cdf_median = self.cdf(params, median_t)
            surv_median = 1.0 - cdf_median
            risk_scores = -torch.log(surv_median.clamp(min=1e-12))

            cdf = self.cdf(params, eval_t)
            surv = 1.0 - cdf
            haz = -torch.log(surv.clamp(min=1e-12))

            haz_fixed = haz.clone()
            haz_fixed[:, 0] = risk_scores.squeeze(-1)
            outputs["cum_hazard_seqs"] = haz_fixed.transpose(0, 1).contiguous()

            t_min = constant_dict["t_min"].to(device)
            t_max = constant_dict["t_max"].to(device)
            num_int_steps = int(constant_dict["NUM_INT_STEPS"])
            grid = torch.linspace(float(t_min), float(t_max), num_int_steps, device=device)
            surv_grid = 1.0 - self.cdf(params, grid.unsqueeze(0).repeat(B, 1))
            outputs["survival_seqs"] = surv_grid.transpose(0, 1).contiguous()

            for eps in (0.1, 0.2, 0.3, 0.4, 0.5):
                key = f"t_max_{eps}"
                if key in constant_dict:
                    t_max_eps = constant_dict[key].to(device)
                    grid_eps = torch.linspace(float(t_min), float(t_max_eps), num_int_steps, device=device)
                    surv_eps = 1.0 - self.cdf(params, grid_eps.unsqueeze(0).repeat(B, 1))
                    outputs[f"survival_seqs_{eps}"] = surv_eps.transpose(0, 1).contiguous()

            outputs["eval_t"] = eval_t[0].detach().cpu()

            if not self._debug_printed:
                self._debug_printed = True
                print(f"DEBUG eval_step: median_t={median_t[0,0]:.4f}, risk min {risk_scores.min():.4f}, max {risk_scores.max():.4f}")

            return outputs

    def predict_step(self, x_dict):
        device = x_dict['x'].device
        with torch.no_grad():
            params, _, _ = self(**x_dict)
            t = torch.arange(0.1, 220.1, 0.1, device=device).unsqueeze(0)
            surv = 1.0 - self.cdf(params, t)
        return {'t': t.squeeze(0), 'p_survival': surv.squeeze(0)}
