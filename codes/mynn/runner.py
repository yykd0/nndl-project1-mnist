import numpy as np
import os

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.train_steps = []
        self.dev_steps = []
        self.global_step = 0

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        validation_freq = kwargs.get("validation_freq", None)
        eval_batch_size = kwargs.get("eval_batch_size", 256)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_score = 0
        best_path = os.path.join(save_dir, 'best_model.pickle')

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            num_batches = int(np.ceil(X.shape[0] / self.batch_size))
            for iteration in range(num_batches):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)
                self.train_steps.append(self.global_step)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                should_evaluate = False
                if validation_freq is not None and self.global_step % validation_freq == 0:
                    should_evaluate = True
                if iteration == num_batches - 1:
                    should_evaluate = True

                if should_evaluate:
                    dev_score, dev_loss = self.evaluate(dev_set, batch_size=eval_batch_size)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    self.dev_steps.append(self.global_step)

                    if dev_score > best_score:
                        self.save_model(best_path)
                        print(f"best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    if len(self.dev_scores) > 0:
                        print(f"[Dev] loss: {self.dev_loss[-1]}, score: {self.dev_scores[-1]}")
        self.best_score = best_score
        self.best_path = best_path

    def evaluate(self, data_set, batch_size=256):
        X, y = data_set
        total_loss = 0.0
        total_correct = 0
        total_num = 0
        for start in range(0, X.shape[0], batch_size):
            batch_X = X[start:start + batch_size]
            batch_y = y[start:start + batch_size]
            logits = self.model(batch_X)
            loss = self.loss_fn(logits, batch_y)
            preds = np.argmax(logits, axis=-1)
            total_loss += loss * batch_X.shape[0]
            total_correct += np.sum(preds == batch_y)
            total_num += batch_X.shape[0]
        return total_correct / total_num, total_loss / total_num

    def predict(self, X, batch_size=256):
        logits_list = []
        for start in range(0, X.shape[0], batch_size):
            logits_list.append(self.model(X[start:start + batch_size]))
        return np.concatenate(logits_list, axis=0)
    
    def save_model(self, save_path):
        self.model.save_model(save_path)
