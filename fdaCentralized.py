from centralized import Centralized
import utils.style_transfer as st


class FdaCentralized(Centralized):

    def __init__(self, args, model, training_dataset, metric, clients=None, b=None, L=None):
        if L is not None or b is not None:
            self.StyleAugment = st.StyleAugment(n_images_per_style=25, size=(1920, 1080), b=b, L=L)
            # return true if the styles are present
            if not self.StyleAugment.load_bank_styles(L=L):
                # if it is false, create the bank for future use
                print('Generate bank of styles...')
                self.StyleAugment.create_bank_styles(clients=clients)
                print('Done.')  
            training_dataset.set_style_tf_fn(self.StyleAugment.apply_style)

        super().__init__(args, model, training_dataset, metric)