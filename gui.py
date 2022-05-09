import PySimpleGUI as pg

pg.theme("DarkAmber")
accuracy = '_'

font = ("Arial", 18)

pg.set_options(font=font)
layout = [
    [pg.Text("Please choose one of the classification methods : \n") ],
    [pg.Radio('LR', 1, enable_events=True, key='LR')],
    [pg.Radio('SVM', 1, enable_events=True, key='SVM')],
    [pg.Radio('DT', 1, enable_events=True, key='DT')],
    [pg.Text("\n") ],
    [pg.Text("Accuracy : " + str(accuracy))],
]


window = pg.Window("Bankrupsy Predector" , layout , size = (600,400))

curr_choice = ""
while True :
    event , values = window.read()

    if (event == pg.WIN_CLOSED) :
        break

    if (event == "LR" and curr_choice != "LR") :
        curr_choice = "LR"
        print("LR")

    elif (event == "SVM" and curr_choice != "SVM") :
        curr_choice = "SVM"
        print("SVM")

    elif (event == "DT" and curr_choice != "DT") :
        curr_choice = "DT"
        print("DT")


window.close()
