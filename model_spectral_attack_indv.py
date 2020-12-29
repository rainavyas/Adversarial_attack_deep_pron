import torch
import torch_dct as dct
from model_deep_pron_noI import Deep_Pron

class Spectral_attack(torch.nn.Module):
    def __init__(self, spectral_dim, mfcc_dim, trained_model_path, init_root):

        super(Spectral_attack, self).__init__()

        self.trained_model_path = trained_model_path

        self.noise_root = torch.nn.Parameter(init_root, requires_grad=True)

        self.spectral_dim = spectral_dim
        self.mfcc_dim = mfcc_dim

    def forward(self, p_vects, q_vects, p_frames_mask, q_frames_mask):
        '''
        p/q_vects = [num_speakers X num_feats X max_num_mfcc_frames x mfcc_dim]
        p/q_frames_mask = [num_speakers X num_feats X max_num_mfcc_frames x mfcc_dim]
                          -> The associated 0s and 1s mask of p/q_lengths
        n.b. mfcc_dim = 13 usually (using c0 for energy instead of log-energy)
             num_feats = 46*47*0.5 = 1128 usually
             max_num_mfcc_frames = the maximum number of frames associated
             with a particular phone for any speaker -> often set to 4000
        '''
        # Apply the attack
        noise = torch.exp(self.noise_root)

        # Need to add spectral noise
        # Pad to spectral dimension
        padding = torch.zeros(p_vects.size(0), p_vects.size(1), p_vects.size(2), self.spectral_dim - self.mfcc_dim)
        padded_p_vects = torch.cat((p_vects, padding), 3)
        padded_q_vects = torch.cat((q_vects, padding), 3)

        # Apply inverse dct
        log_spectral_p = dct.idct(padded_p_vects)
        log_spectral_q = dct.idct(padded_q_vects)

        # Apply inverse log
        spectral_p = torch.exp(log_spectral_p)
        spectral_q = torch.exp(log_spectral_q)

        # Restructure noise
        noise_struct = noise.unsqueeze(1).unsqueeze(1).repeat(1, p_vects.size(1), p_vects.size(2), 1)

        # Add the adversarial attack noise
        attacked_spectral_p = spectral_p + noise_struct
        attacked_spectral_q = spectral_q + noise_struct

        # Apply the log
        attacked_log_spectral_p = torch.log(attacked_spectral_p)
        attacked_log_spectral_q = torch.log(attacked_spectral_q)

        # Apply the dct
        attacked_padded_p = dct.dct(attacked_log_spectral_p)
        attacked_padded_q = dct.dct(attacked_log_spectral_q)

        # Truncate to mfcc dimension
        p_vects_attacked = torch.narrow(attacked_padded_p, 3, 0, self.mfcc_dim)
        q_vects_attacked = torch.narrow(attacked_padded_q, 3, 0, self.mfcc_dim)

        # Apply mask of zeros/ones, to ensure spectral noise only applied up to p/q lengths
        p_vects_masked = p_vects_attacked * p_frames_mask
        q_vects_masked = q_vects_attacked * q_frames_mask


        # Pass through trained model
        trained_model = torch.load(self.trained_model_path)
        trained_model.eval()
        y = trained_model(p_vects_masked, q_vects_masked, p_frames_mask, q_frames_mask)

        return y

    def get_noise(self):
        '''
        return the spectral noise vector
        '''
        return torch.exp(self.noise_root)
