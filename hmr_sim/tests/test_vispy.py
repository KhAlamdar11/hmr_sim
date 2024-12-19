from vispy import app, gloo

canvas = app.Canvas(keys='interactive')

@canvas.connect
def on_draw(event):
    gloo.clear(color='blue')

canvas.show()
app.run()
