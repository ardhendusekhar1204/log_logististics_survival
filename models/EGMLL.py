
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
