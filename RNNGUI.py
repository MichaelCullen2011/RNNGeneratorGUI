import tkinter as tk
from tkinter import *
import hp
import sp

'''
GUI Windows
'''
# Main Window
root = tk.Tk()
canvas = tk.Canvas(root, height=800, width=800, bg='#263D42')
canvas.pack()
# Frame inside main window
frame1 = tk.Frame(root, bg='white')
frame1.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
text = tk.Label(frame1, text='Pick a Model', bg='white', fg='black')
text.place(x=300, y=300)

'''
Initialising Logic
'''
HP = False
SP = False
Train = False
Generate = False

'''
Functions
'''


def reset_model():
    # Main Window
    canvas = tk.Canvas(root, height=800, width=800, bg='#263D42')
    canvas.place()
    # Frame inside main window
    frame1 = tk.Frame(root, bg='white')
    frame1.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    train.place(x=350, y=750)
    generate.place(x=500, y=750)
    resetModel.place(x=700, y=30)
    text = tk.Label(frame1, text='Pick a Model', bg='white', fg='black')
    text.place(x=300, y=300)
    train['state'] = DISABLED
    generate['state'] = DISABLED


def choose_file():
    frame2 = tk.Frame(root, bg='#5CC7B2')
    frame2.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    shakes = tk.Button(root, text='Shakespeare', padx=10, pady=5,
                       fg='white', bg='#263D42', command=shakespeare, state=NORMAL)
    shakes.place(x=250, y=200)

    harry = tk.Button(root, text='Harry Potter', padx=10, pady=5,
                      fg='white', bg='#263D42', command=harry_potter, state=NORMAL)
    harry.place(x=450, y=200)


def harry_potter():
    global HP, SP
    frame2 = tk.Frame(root, bg='gray')
    frame2.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    SP = False
    HP = True
    sp.run = False
    hp.run = True
    train['state'] = NORMAL
    generate['state'] = NORMAL
    run['state'] = NORMAL
    harry_text = tk.Label(root, text='Harry Potter Model', bg='gray', fg='white')
    harry_text.place(x=350, y=100)
    if Train:
        train_text = tk.Label(root, text='Train On', bg='gray', fg='white')
        train_text.place(x=100, y=200)
    elif not Train:
        train_text = tk.Label(root, text='Train Off', bg='gray', fg='white')
        train_text.place(x=100, y=200)
    if Generate:
        train_text = tk.Label(root, text='Generate On', bg='gray', fg='white')
        train_text.place(x=300, y=200)
    elif not Generate:
        train_text = tk.Label(root, text='Generate Off', bg='gray', fg='white')
        train_text.place(x=300, y=200)


def shakespeare():
    global HP, SP
    frame2 = tk.Frame(root, bg='gray')
    frame2.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    HP = False
    SP = True
    hp.run = False
    sp.run = True
    train['state'] = NORMAL
    generate['state'] = NORMAL
    run['state'] = NORMAL
    sp_text = tk.Label(root, text='Shakespeare Model', bg='gray', fg='white')
    sp_text.place(x=350, y=100)
    if Train:
        train_text = tk.Label(root, text='Train On', bg='gray', fg='white')
        train_text.place(x=100, y=200)
    elif not Train:
        train_text = tk.Label(root, text='Train Off', bg='gray', fg='white')
        train_text.place(x=100, y=200)
    if Generate:
        train_text = tk.Label(root, text='Generate On', bg='gray', fg='white')
        train_text.place(x=300, y=200)
    elif not Generate:
        train_text = tk.Label(root, text='Generate Off', bg='gray', fg='white')
        train_text.place(x=300, y=200)


def train():
    global HP, SP, Train
    if HP:
        if hp.Train:
            hp.Train = False
            hp.Load = True
        else:
            hp.Train = True
            hp.Load = False
        Train = hp.Train
    elif SP:
        if sp.Train:
            sp.Train = False
            sp.Load = True
        else:
            sp.Train = True
            sp.Load = False
        Train = sp.Train
    if Train:
        train_text = tk.Label(root, text='Train On', bg='gray', fg='white')
        train_text.place(x=100, y=200)
    elif not Train:
        train_text = tk.Label(root, text='Train Off', bg='gray', fg='white')
        train_text.place(x=100, y=200)


def generate():
    global HP, SP, Generate
    Generate = True
    if HP:
        if hp.Generate:
            hp.Generate = False
        else:
            hp.Generate = True
        Generate = hp.Generate
    elif SP:
        if sp.Generate:
            sp.Generate = False
        else:
            sp.Generate = True
        Generate = sp.Generate
    if Generate:
        train_text = tk.Label(root, text='Generate On', bg='gray', fg='white')
        train_text.place(x=300, y=200)
    elif not Generate:
        train_text = tk.Label(root, text='Generate Off', bg='gray', fg='white')
        train_text.place(x=300, y=200)


def run_program():
    frame1 = tk.Frame(root, bg='cyan')
    frame1.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    run_text = tk.Label(frame1, text='Running', bg='cyan', fg='black')
    run_text.place(x=300, y=100)
    if HP:
        hp.start()
        hp_text = tk.Label(frame1, text='Harry Potter Model', bg='cyan', fg='black')
        hp_text.place(x=300, y=200)
        y_train = 250
        hp.training_model()
        # if hp.training_step:
        #     training_text = tk.Label(frame1, text=text1, bg='white', fg='black')
        #     training_text.place(x=100, y=y_train)
        #     training_text = tk.Label(frame1, text=text2, bg='white', fg='black')
        #     training_text.place(x=100, y=y_train)
        #     y_train += 50
        hp.loading_model()
        if hp.loading_step:
            loading_text = tk.Label(root, text='Loading...', bg='white', fg='black')
            loading_text.place(x=300, y=400)
        hp.generate_step()
        if hp.generate_step:
            generate_text = tk.Label(root, text='Generating...', bg='white', fg='black')
            generate_text.place(x=300, y=500)
    elif SP:
        sp.start()
        sp_text = tk.Label(root, text='Shakespeare Model', bg='white', fg='black')
        sp_text.place(x=300, y=200)
        if sp.training_step:
            training_text = tk.Label(root, text='Training...', bg='white', fg='black')
            training_text.place(x=300, y=300)
        if sp.loading_step:
            loading_text = tk.Label(root, text='Loading...', bg='white', fg='black')
            loading_text.place(x=300, y=400)
        if sp.generate_step:
            generate_text = tk.Label(root, text='Generating...', bg='white', fg='black')
            generate_text.place(x=300, y=500)


'''
Buttons
'''
chooseFile = tk.Button(root, text='Choose File', padx=10, pady=5,
                               fg='white', bg='#263D42', command=choose_file, state=NORMAL)
chooseFile.place(x=150, y=750)

train = tk.Button(root, text='Train', padx=10, pady=5, fg='white', bg='#263D42',
                          command=train, state=DISABLED)
train.place(x=350, y=750)

generate = tk.Button(root, text='Generate Text', padx=10, pady=5, fg='white', bg='#263D42',
                             command=generate, state=DISABLED)
generate.place(x=500, y=750)

run = tk.Button(root, text='Run', padx=10, pady=5,
                        fg='white', bg='#263D42', command=run_program, state=DISABLED)
run.place(x=700, y=750)

resetModel = tk.Button(root, text='Reset Model', padx=10, pady=5,
                               fg='white', bg='#263D42', command=reset_model, state=NORMAL)
resetModel.place(x=700, y=30)

'''
Main Loop
'''
root.mainloop()






