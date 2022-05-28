from locale import normalize
from math import cos, sin
import os, sys
from pickletools import stringnl_noescape
from turtle import Screen
import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr

"""
Goal: Get Global Lighting (Sun) in Scene
"""

class ConnectedComponent:

    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.toastiness = 0
        self.selfRotationAngle = 0
        self.parent = None
        self.child = None
        self.modelTransform = pyrr.matrix44.create_identity()
    
    def updateTransform(self):

        self.modelTransform = pyrr.matrix44.create_identity()

        #first apply local transformations, gets object into position on parent etc

        #standard pitch, yaw and roll
        self.modelTransform = pyrr.matrix44.multiply(
            m1 = self.modelTransform,
            m2 = pyrr.matrix44.create_from_eulers(
                eulers = np.radians(self.eulers),
                dtype = np.float32
            )
        )

        #spin object
        if self.selfRotationAngle != 0:

            x = sin(np.radians(self.eulers[0])) * sin(np.radians(self.eulers[2]))
            y = sin(np.radians(self.eulers[0])) * cos(np.radians(self.eulers[2]))
            z = cos(np.radians(self.eulers[0]))

            C=cos(self.selfRotationAngle)
            S=sin(self.selfRotationAngle)
            t=1-C

            selfRotationTransform = np.mat([
                t*x*x+C,    t*x*y-S*z,  t*x*z+S*y,  0,
                t*x*y+S*z,  t*y*y+C,    t*y*z-S*x,  0,
                t*x*z-S*y,  t*y*z+S*x,  t*z*z+C,    0,
                0,          0,          0,          1,
            ]).reshape(4,4)

            self.modelTransform = pyrr.matrix44.multiply(
                m1 = self.modelTransform,
                m2 = selfRotationTransform
            )        

        #translate object
        self.modelTransform = pyrr.matrix44.multiply(
            m1 = self.modelTransform,
            m2 = pyrr.matrix44.create_from_translation(
                vec = self.position,
                dtype = np.float32
            )
        )

        #now apply the parent's transformation
        if self.parent is not None:
            self.modelTransform = pyrr.matrix44.multiply(
                m1 = self.modelTransform,
                m2 = self.parent.modelTransform
            )
        
        #then trigger the child object to update its transform
        if self.child is not None:
            self.child.updateTransform()


class Scene:

    def __init__(self, position, eulers):

        self.position = np.array(position, dtype = np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        
        self.stones = ConnectedComponent(
            position = [position[0], position[1], position[2]], 
            eulers = [eulers[0], eulers[1], eulers[2]]
        )
        self.logs = ConnectedComponent(
            position = [position[0], position[1], position[2]], 
            eulers = [eulers[0], eulers[1], eulers[2]]
        )
        self.floor = ConnectedComponent(
            position = [position[0], position[1], position[2]], 
            eulers = [eulers[0], eulers[1], eulers[2]]
        )
        self.stick = ConnectedComponent(
            position = [-5, 20, 10],
            eulers = [eulers[0]+270, eulers[1], eulers[2]]
        )
        self.marshmallow = ConnectedComponent(
            position = [-1, 0, 10], 
            eulers = [eulers[0], eulers[1], eulers[2]+17]
        )

        self.marshmallow.parent = self.stick
        self.stick.child = self.marshmallow

        self.logs.parent = self.stones
        self.stones.child = self.logs
 
    def update(self):

        # update
        self.stick.updateTransform()

        # cook marshmallow proportional to inverse square of distance
        distance = np.linalg.norm(np.dot(np.array([0, 0, 0, 1]), self.marshmallow.modelTransform))

        # stop toasting after a certain distance
        if distance < 13:
            self.marshmallow.toastiness += 0.5*(distance**-2)

        # max toastiness
        if self.marshmallow.toastiness > 100:
            self.marshmallow.toastiness = 100




class App:

    def __init__(self, screenWidth, screenHeight):

        os.chdir(os.path.dirname(sys.argv[0]))

        self.screenWidth = screenWidth
        self.screenHeight = screenHeight

        self.renderer = GraphicsEngine()

        self.scene = Scene(
            position = [0,0,0],
            eulers = [0,0,0]
        )

        self.lastTime = pg.time.get_ticks()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0
        self.lightCount = 0

        self.mainLoop()

    def mainLoop(self):
        running = True
        while (running):
            #check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
            
            self.handleKeys()

            self.scene.update()
            
            self.renderer.render(self.scene)

            #timing
            self.calculateFramerate()
        self.quit()

    def handleKeys(self):

        keys = pg.key.get_pressed()

        rotrate = self.frameTime / 16
        disprate = self.frameTime / 32

        if keys[pg.K_w]:
            self.scene.stick.position[1] -= disprate
            if self.scene.stick.position[1] < 10:
                self.scene.stick.position[1] = 10

        elif keys[pg.K_s]:
            self.scene.stick.position[1] += disprate
            if self.scene.stick.position[1] > 20:
                self.scene.stick.position[1] = 20
                
        elif keys[pg.K_RIGHT]:
            self.scene.stick.eulers[2] += rotrate
            if self.scene.stick.eulers[2] > 360:
                self.scene.stick.eulers[2] -= 360            
            if self.scene.stick.eulers[2] > 45 and self.scene.stick.eulers[2] < 315:
                self.scene.stick.eulers[2] = 45

        elif keys[pg.K_LEFT]:
            self.scene.stick.eulers[2] -= rotrate
            if self.scene.stick.eulers[2] < 0:
                self.scene.stick.eulers[2] += 360
            if self.scene.stick.eulers[2] < 315 and self.scene.stick.eulers[2] > 45:
                self.scene.stick.eulers[2] = 315

        elif keys[pg.K_DOWN]:
            self.scene.stick.eulers[0] -= rotrate
            if self.scene.stick.eulers[0] < 225:
                self.scene.stick.eulers[0] = 225

        elif keys[pg.K_UP]:
            self.scene.stick.eulers[0] += rotrate
            if self.scene.stick.eulers[0] > 315:
                self.scene.stick.eulers[0] = 315

        elif keys[pg.K_SPACE]:
            self.scene.stick.selfRotationAngle += 0.1*rotrate
            

    def calculateFramerate(self):

        self.currentTime = pg.time.get_ticks()
        delta = self.currentTime - self.lastTime
        if (delta >= 1000):
            framerate = max(1,int(1000.0 * self.numFrames/delta))
            pg.display.set_caption(f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(1,framerate))
        self.numFrames += 1

    def quit(self):
        
        self.renderer.destroy()


class GraphicsEngine:

    def __init__(self):

        self.palette = {
            "SkyBlue": np.array([132/255,206/255,235/255], dtype = np.float32)
        }

        #initialise pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640,480), pg.OPENGL|pg.DOUBLEBUF)

        #initialise opengl
        glClearColor(self.palette["SkyBlue"][0], self.palette["SkyBlue"][1], self.palette["SkyBlue"][2], 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #create renderpasses and resources
        shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        marshmallowShader = self.createShader("shaders/vertexMarshmallow.txt", "shaders/fragmentMarshmallow.txt")

        self.renderPass = RenderPass(shader, marshmallowShader)

        self.stonesTexture = Material("gfx/stone.jpg")
        self.stonesMesh = Mesh("models/stones.obj")

        self.logsTexture = Material("gfx/dark_wood.jpg")
        self.logsMesh = Mesh("models/logs.obj")

        self.floorTexture = Material("gfx/grass.jpg")
        self.floorMesh = Mesh("models/plane.obj")

        self.stickTexture = Material("gfx/light_wood.jpg")
        self.stickMesh = Mesh("models/stick.obj")

        self.marshmallowTexture = Material("gfx/white.jpg")
        self.marshmallowMesh = Mesh("models/marshmallow.obj")
    
    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath,'r') as f:
            fragment_src = f.readlines()
        
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        
        return shader

    def render(self, scene):

        #refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.renderPass.render(scene, self)

        pg.display.flip()

    def destroy(self):

        self.stonesTexture.destroy()
        self.stonesMesh.destroy()
        self.logsTexture.destroy()
        self.logsMesh.destroy()
        self.stickTexture.destroy()
        self.stickMesh.destroy()
        self.marshmallowTexture.destroy()
        self.marshmallowMesh.destroy()
        self.renderPass.destroy()
        pg.quit()


class RenderPass:

    def __init__(self, shader, marshmallowShader):

        # initialise opengl
        self.shader = shader
        self.marshmallowShader = marshmallowShader

        # sun colour and direction
        sunColor = np.array([1,1,1], dtype=np.float32)
        sunDirection = pyrr.vector.normalize(np.array([1,0,-1], dtype=np.float32))

        # projection transform
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 800/600, 
            near = 0.1, far = 100, dtype=np.float32
        )        
        
        # initialise shader and get matrix memory locations
        glUseProgram(self.shader)

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader,"projection"),
            1, GL_FALSE, projection_transform
        )

        glUniform3fv(glGetUniformLocation(self.shader,"sunColor"), 1, sunColor)
        glUniform3fv(glGetUniformLocation(self.shader,"sunDirection"), 1, sunDirection)

        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.textureScaleLocation = glGetUniformLocation(self.shader, "textureScale")

        # initialise marshmallow shader and get matrix memory locations
        glUseProgram(self.marshmallowShader)

        glUniformMatrix4fv(
            glGetUniformLocation(self.marshmallowShader,"projection"),
            1, GL_FALSE, projection_transform
        )        

        glUniform3fv(glGetUniformLocation(self.marshmallowShader,"sunColor"), 1, sunColor)
        glUniform3fv(glGetUniformLocation(self.marshmallowShader,"sunDirection"), 1, sunDirection)

        self.marshmallowModelMatrixLocation = glGetUniformLocation(self.marshmallowShader, "model")
        self.marshmallowViewMatrixLocation = glGetUniformLocation(self.marshmallowShader, "view")
        self.marshmallowColorLoc = glGetUniformLocation(self.marshmallowShader, "objectColor")

    def render(self, scene, engine):

        glUseProgram(self.shader)

        view_transform = pyrr.matrix44.create_look_at(
            eye = np.array([0,32,20], dtype = np.float32),
            target = np.array([0,0,10], dtype = np.float32),
            up = np.array([0,0,1], dtype = np.float32), dtype = np.float32
        )

        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)
        
        #stones
        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, scene.stones.modelTransform)
        glUniform1f(self.textureScaleLocation, 0.2)
        engine.stonesTexture.use()
        glBindVertexArray(engine.stonesMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.stonesMesh.vertex_count)

        #logs
        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, scene.logs.modelTransform)
        glUniform1f(self.textureScaleLocation, 1)
        engine.logsTexture.use()
        glBindVertexArray(engine.logsMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.logsMesh.vertex_count)

        #stick
        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, scene.stick.modelTransform)
        glUniform1f(self.textureScaleLocation, 1)
        engine.stickTexture.use()
        glBindVertexArray(engine.stickMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.stickMesh.vertex_count)

        #floor
        glUniform1f(self.textureScaleLocation, 10)
        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, scene.floor.modelTransform)
        engine.floorTexture.use()
        glBindVertexArray(engine.floorMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.floorMesh.vertex_count)

        glUseProgram(self.marshmallowShader)
        glUniformMatrix4fv(self.marshmallowViewMatrixLocation, 1, GL_FALSE, view_transform)

        #marshmallow
        glUniformMatrix4fv(self.marshmallowModelMatrixLocation, 1, GL_FALSE, scene.marshmallow.modelTransform)
        (r,g,b) = self.marshmallowBaseColor(scene)
        glUniform3fv(self.marshmallowColorLoc, 1, np.array([r/255, g/255, b/255], dtype = np.float32))
        glBindVertexArray(engine.marshmallowMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.marshmallowMesh.vertex_count)

    def marshmallowBaseColor(self, scene):

        toastiness = scene.marshmallow.toastiness

        if toastiness < 35:
            r = 255
            g = (-64 * (toastiness/35)) + 255
            b = (-208 * (toastiness/35)) + 255
        elif toastiness < 65:
            relative = toastiness - 35
            r = 255
            g = (-15 * (relative/30)) + 191
            b = (-42 * (relative/30)) + 47
        else:
            relative = toastiness - 65
            r = (-250 * (relative/35)) + 255
            g = (-171 * (relative/35)) + 176
            b = 5

        return (r,g,b)

    def destroy(self):

        glDeleteProgram(self.shader)
        glDeleteProgram(self.marshmallowShader)


class Mesh:

    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
    
    def loadMesh(self, filename):

        #raw, unassembled data
        v = []
        vt = []
        vn = []
        
        #final, assembled and packed result
        vertices = []

        #open the obj file and read the data
        with open(filename,'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag=="v":
                    #vertex
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag=="vt":
                    #texture coordinate
                    line = line.replace("vt ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag=="vn":
                    #normal
                    line = line.replace("vn ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag=="f":
                    #face, three or more vertices in v/vt/vn form
                    line = line.replace("f ","")
                    line = line.replace("\n","")
                    #get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        #break out into [v,vt,vn],
                        #correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))


class Material:
    
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert_alpha()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))

myApp = App(800,600)