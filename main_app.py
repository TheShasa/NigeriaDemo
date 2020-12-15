import tkinter as tk

from audio import record_audio
import os
from datetime import datetime
import pandas as pd
from model import vggvox_model
import constants as c
from scoring import get_embedding
import numpy as np
from scipy.spatial.distance import cdist


class Main(tk.Frame):
    INRECORD = 'Recording...'
    WAIT = 'Waiting for a press'
    DB_ADD = 'Added to Database'

    RECORD_SECODNS = 3  # 5
    FOLDER = "DEMO"

    def __init__(self, root):
        super().__init__(root)

        self.db_file = os.path.join(self.FOLDER, 'enroll_list.csv')
        self.embed_file = os.path.join(self.FOLDER, 'embed.npy')
        self.db = pd.read_csv(self.db_file)

        self.root = root
        self.status = self.WAIT
        self.lastname = ''
        self.last_embed = None
        self.model = self.load_model()
        self.embeding = self.load_embed()

        print(self.db.shape, self.embeding.shape)
        self.init_main()
        self.pack(fill=tk.BOTH, expand=1)

    def init_main(self):
        self.name_text = tk.Label(self, text="Specify your name:")
        self.name_text.place(relx=0.3, rely=0.1, anchor='n')

        self.name_edit = tk.Entry(self, text="")
        self.name_edit.place(relx=0.7, rely=0.1, relheight=0.1, relwidth=0.3, anchor='n')

        self.start_button = tk.Button(self, text="Start Record", compound=tk.TOP, command=self.start_record)
        self.start_button.place(relx=0.5, rely=0.3, relwidth=0.7, anchor='n')

        # self.stop_button = tk.Button(root, text="Stop Record", compound=tk.TOP, command=self.stop_record)
        # self.stop_button.place(relx=0.7, rely=0.3, anchor='n')

        self.status_text = tk.Label(self, text=self.status)
        self.status_text.place(relx=0.5, rely=0.5, anchor='n')

        self.db_button = tk.Button(self, text="Add last record in DB", compound=tk.TOP, command=self.add_to_db)
        self.db_button.place(relx=0.5, rely=0.7, relwidth=0.7, anchor='n')

        self.db_button_reset = tk.Button(self, text="Reset the DB", compound=tk.TOP, command=self.reset_db)
        self.db_button_reset.place(relx=0.5, rely=0.8, relwidth=0.7, anchor='n')
        self.db_button_predict = tk.Button(self, text="Predict the name", compound=tk.TOP, command=self.predict)
        self.db_button_predict.place(relx=0.5, rely=0.9, relwidth=0.7, anchor='n')
        if not os.path.exists(os.path.join(self.FOLDER, 'all_records')):
            os.mkdir(os.path.join(self.FOLDER, 'all_records'))

    def load_model(self):
        model = vggvox_model()
        model.load_weights("data/model/weights.h5")
        return model

    def load_embed(self):
        if os.path.exists(self.embed_file):
            return np.load(self.embed_file)
        return np.array([])

    def predict(self):

        distances = pd.DataFrame(cdist(self.embeding, self.last_embed, metric=c.COST_METRIC))

        print(distances)
        min_speaker = distances.iloc[:, 0].argmin()

        print(self.db['speaker'][min_speaker])
        self.update_status('Predicted ' + self.db['speaker'][min_speaker] + '!')

    def reset_db(self):

        self.update_status('Reset the Database')
        db_back = self.db.copy()

        db_back.to_csv(os.path.join(self.FOLDER, 'enroll_list_back.csv'), index=False)
        self.db = self.db.iloc[:0]
        self.db.to_csv(self.db_file, index=False)

        np.save(os.path.join(self.FOLDER, 'embed_back_back.npy'), self.embeding)

        self.embeding = np.array([])
        np.save(self.embed_file, self.embeding)

    def add_to_db(self):
        speaker = self.name_edit.get()
        if speaker != "" and self.lastname != "":
            self.update_status(speaker + ' ' + self.DB_ADD)
            print('Add to db')
            self.db = self.db.append({'filename': self.lastname, 'speaker': speaker},
                                     ignore_index=True)

            self.embeding = np.vstack([self.embeding, self.last_embed]) if self.embeding.size else self.last_embed

            print(self.embeding.shape)

            self.db.to_csv(self.db_file, index=False)

            np.save(self.embed_file, self.embeding)

            print(self.db.head())
        else:
            self.update_status('Specify the name!')
            print('Specify name')
        pass

    def start_record(self):

        if self.status == self.INRECORD:
            print("We are in record!")
            return

        # current date and time
        now = datetime.now()

        timestamp = datetime.timestamp(now)
        print("timestamp =", timestamp)
        self.update_status(self.INRECORD)
        self.start_button.configure(foreground='red')
        self.root.update()

        self.lastname = os.path.join(self.FOLDER, 'all_records', str(timestamp) + '.wav')
        record_audio(self.RECORD_SECODNS, self.lastname, self.root)

        self.start_button.configure(foreground='black')
        self.update_status('Loading...')

        self.last_embed = np.array(get_embedding(self.model, self.lastname, c.MAX_SEC)).reshape(1, -1)
        self.update_status('Recorded!')
        return

    def update_status(self, newstatus):
        print(newstatus)
        self.status = newstatus
        self.status_text.config(text=self.status)
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = Main(root)
    # app.pack()
    root.title("Voice Demo")
    root.geometry("250x250+0+0")
    root.resizable(True, True)
    root.mainloop()
