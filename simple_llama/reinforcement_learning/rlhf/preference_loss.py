import torch


class PreferenceLoss:
    def __init__(self, beta: float):
        self.beta = beta

    def calculate_loss(self, theta_logits: torch.Tensor, ref_logits: torch.Tensor, target_ids: torch.Tensor,
                       pad_token: int, device: str) -> torch.Tensor:
        """
        Very simple preference loss based on DPO paper and references the implementation from HF's trainer file
        Can be found here:
        DPO Loss: https://arxiv.org/pdf/2305.18290
        HF Trainer: https://github.com/huggingface/trl/blob/main/trl/trainer/reward_trainer.py#L265

        :param theta_logits: raw logits from the current LM undergoing RLHF, should be of shape (batch, seq_len, vocab_size)
        :param ref_logits: raw logits from the frozen sft LM, also of shape (batch, seq_len, vocab_size)
        :param target_ids: tensor of shape (batch, seq_len), should contain target idx of targeted responses. Prompt should be padded out
        :param pad_token: int id of the padding token, acts as an implicit mask
        :param device: 'cpu' or 'cuda' (or any other valid devices)
        :return: a single scalar tensor, representing the loss value
        """
        # Basic assertions to accomodate assumptions
        assert theta_logits.shape[0] % 2 == 0, (f"Expected even number of batch size for accepted/rejected pairs, "
                                                f"instead got {theta_logits.shape=}")
        assert len(theta_logits.shape) == 3, (f"Expected theta_logits to be of shape (batch, seq_len, vocab_size) "
                                              f"instead got {theta_logits.shape=}")
        assert theta_logits.shape == ref_logits.shape
        assert len(target_ids.shape) == 2 and target_ids.shape == theta_logits.shape[:2]


        # Transform into log prob and gather based on target ids
        # unsqueeze target_ids turns it from (B, T) -> (B, T, 1) so that dimensions line up
        # squeeze the final _log_prob tensor would transform it back from (B, T, 1) to (B, T)
        theta_log_prob = torch.nn.functional.log_softmax(theta_logits, dim=-1)
        theta_log_prob = theta_log_prob.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        ref_log_prob = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        ref_log_prob = ref_log_prob.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # Create and apply mask. Mask will be made up of 0s and 1s
        mask = torch.tensor(target_ids != pad_token, dtype=torch.float32, device=device)
        assert theta_log_prob.shape == ref_log_prob.shape == target_ids.shape == mask.shape, "Shouldn't occur"

        theta_log_prob = theta_log_prob * mask
        ref_log_prob = ref_log_prob * mask

        # Find per-response token contribution to normalize, preventing bias against longer sequences
        mask_len = mask.sum(dim=1).clamp(min=1)

        # Extract accepted abd rejected logits, then sum along the seq_len (T) dim to get the log prob for that response
        accepted_theta = theta_log_prob[0::2].sum(dim=1) / mask_len[0::2]
        rejected_theta = theta_log_prob[1::2].sum(dim=1) / mask_len[1::2]
        accepted_ref = ref_log_prob[0::2].sum(dim=1) / mask_len[0::2]
        rejected_ref = ref_log_prob[1::2].sum(dim=1) / mask_len[1::2]


        # The following is based on Equation 7 in DPO paper
        # -----------------------------

        # log(pi_theta(y_w | x) / pi_ref(y_w | x)) and log(pi_theta(y_l | x) / pi_ref(y_l | x)) respectively
        scaled_log_winner_ratio = accepted_theta - accepted_ref
        scaled_log_loser_ratio = rejected_theta - rejected_ref

        loss = -torch.nn.functional.logsigmoid(self.beta * (scaled_log_winner_ratio - scaled_log_loser_ratio)).mean()
        assert list(loss.shape) == [], "Shouldn't occur"  # Final loss should be a scalar

        return loss
