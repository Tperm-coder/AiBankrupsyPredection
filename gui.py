import PySimpleGUI as pg

file = open("results.txt" , 'r')
data = file.read()
file.close()

data = data.split('\n')
for i in range(len(data)) :
    data[i] = data[i].split(',')

LRb = data[0][0]
LRa = data[0][1]
LRCF1 = data[0][2]
LRCF2 = data[0][3]
LRCF3 = data[0][4]
LRCF4 = data[0][5]

DTb = data[1][0]
DTa = data[1][1]
DTCF1 = data[1][2]
DTCF2 = data[1][3]
DTCF3 = data[1][4]
DTCF4 = data[1][5]

SVMb = data[2][0]
SVMa = data[2][1]
SVMCF1 = data[2][2]
SVMCF2 = data[2][3]
SVMCF3 = data[2][4]
SVMCF4 = data[2][5]

pg.theme("DarkAmber")

accuracyb = '_'
accuracya = '_'

font = ("Arial", 18)

pg.set_options(font=font)
layout = [
    [pg.Text("Please choose one of the classification methods : \n") ],
    [pg.Radio('LR', 1, enable_events=True, key='LR'),pg.Text("                   ") , pg.Text("Confusion matrix")],
    [pg.Radio('SVM', 1, enable_events=True, key='SVM'),pg.Text("                     ") ,pg.Text('_' , key = "CF1") , pg.Text('_' , key = "CF2") ],
    [pg.Radio('DT', 1, enable_events=True, key='DT') , pg.Text("                        ") , pg.Text('_' , key = "CF3") , pg.Text('_' , key = "CF4")],
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

        window['CF1'].update(value = LRCF1)
        window['CF2'].update(value = LRCF2)
        window['CF3'].update(value = LRCF3)
        window['CF4'].update(value = LRCF4)


    elif (event == "SVM" and curr_choice != "SVM") :
        curr_choice = "SVM"
        window['accuracyb'].update(value = "Accuracy before normalization: " + str(SVMb))
        window['accuracya'].update(value = "Accuracy after normalization: " + str(SVMa))

        window['CF1'].update(value = SVMCF1)
        window['CF2'].update(value = SVMCF2)
        window['CF3'].update(value = SVMCF3)
        window['CF4'].update(value = SVMCF4)


    elif (event == "DT" and curr_choice != "DT") :
        curr_choice = "DT"
        window['accuracyb'].update(value = "Accuracy before normalization: " + str(DTb))
        window['accuracya'].update(value = "Accuracy after normalization: " + str(DTa))

        window['CF1'].update(value = DTCF1)
        window['CF2'].update(value = DTCF2)
        window['CF3'].update(value = DTCF3)
        window['CF4'].update(value = DTCF4)




window.close()
