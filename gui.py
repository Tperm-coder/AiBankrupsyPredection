import PySimpleGUI as pg

file = open("results.txt" , 'r')
data = file.read()
file.close()

data = data.split('\n')
for i in range(len(data)) :
    data[i] = data[i].split(',')

LRb = data[0][0]
LRa = data[0][1]

DTb = data[1][0]
DTa = data[1][1]

SVMb = data[2][0]
SVMa = data[2][1]

pg.theme("DarkAmber")

accuracyb = '_'
accuracya = '_'

font = ("Arial", 18)

pg.set_options(font=font)
layout = [
    [pg.Text("Please choose one of the classification methods : \n") ],
    [pg.Radio('LR', 1, enable_events=True, key='LR')],
    [pg.Radio('SVM', 1, enable_events=True, key='SVM')],
    [pg.Radio('DT', 1, enable_events=True, key='DT')],
    [pg.Text("\n") ],
    [pg.Text("Accuracy before normalization: " + str(accuracyb) , key = 'accuracyb')],
    [pg.Text("Accuracy after normalization: " + str(accuracya) , key = 'accuracya')],
]


window = pg.Window("Bankrupsy Predector" , layout , size = (600,400))

curr_choice = ""
while True :
    event , values = window.read()

    if (event == pg.WIN_CLOSED) :
        break

    if (event == "LR" and curr_choice != "LR") :
        curr_choice = "LR"
        window['accuracyb'].update(value = "Accuracy before normalization: " + str(LRb))
        window['accuracya'].update(value = "Accuracy after normalization: " + str(LRa))

    elif (event == "SVM" and curr_choice != "SVM") :
        curr_choice = "SVM"
        window['accuracyb'].update(value = "Accuracy before normalization: " + str(SVMb))
        window['accuracya'].update(value = "Accuracy after normalization: " + str(SVMa))


    elif (event == "DT" and curr_choice != "DT") :
        curr_choice = "DT"
        window['accuracyb'].update(value = "Accuracy before normalization: " + str(DTb))
        window['accuracya'].update(value = "Accuracy after normalization: " + str(DTa))



window.close()
