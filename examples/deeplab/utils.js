class Utils {
  constructor(canvas) {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.progressCallback;
    this.modelFile;
    this.labelsFile;
    this.inputSize = [257, 257, 3];
    this.outputSize = [257, 257, 1];
    this.preOptions;
    this.postOptions;
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2');
    this.gl_utils = new WebGLUtils(this.gl);
    this.preprocessShader = null;

    this.initialized = false;

    this.setupPreprocessShader();
  }

  async init(backend, prefer) {
    this.initialized = false;
    let result;
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      printTfLiteModel(this.tfModel);
    }
    let kwargs = {
      rawModel: this.tfModel,
      backend: backend,
      prefer: prefer,
    };
    this.model = new TFliteModelImporter(kwargs);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict() {
    if (!this.initialized) return;
    let start = performance.now();
    let outputTextures = [];
    await this.model.computeInGPU(this.inputTensor, outputTextures);
    let elapsed = performance.now() - start;
    return {
      time: elapsed,
      segMap: {
        data: outputTextures[0],
        outputShape: this.outputSize,
        labels: this.labels,
      },
    };
  }

  async loadModelAndLabels(modelUrl, labelsUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(labelsUrl);
    return {bytes: bytes, text: text};
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      let request = new XMLHttpRequest();
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        if (request.readyState === 4) {
          if (request.status === 200) {
              resolve(request.response);
          } else {
              reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
          }
        }
      };
      if (progress && typeof this.progressCallback !== 'undefined')
        request.onprogress = this.progressCallback;

      request.send();
    });
  }

  getFittedResolution(aspectRatio) {
    const height = this.inputSize[0];
    const width = this.inputSize[1];

    // aspectRatio = width / height
    if (aspectRatio > 1) {
      return [width, Math.floor(height / aspectRatio)];
    } else {
      return [Math.floor(width / aspectRatio), height];
    }
  }


  setupPreprocessShader() {

    const vs =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;

      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
      }`;

    const fs =
      `#version 300 es
      precision highp float;
      out vec4 out_color;

      uniform sampler2D u_image;

      in vec2 v_texcoord;

      void main() {
        out_color = texture(u_image, v_texcoord) / 127.5 - 1.0;
      }`;

    this.preprocessShader = new Shader(this.gl, vs, fs);
    this.preprocessShader.use();
    // this.preprocessShader.set1i('u_length', this._colorPalette.length / 3);

    this.gl_utils.createAndBindTexture({
      name: 'image',
      filter: this.gl.LINEAR,
    });

    this.gl_utils.createTexInFrameBuffer('preprocessResult',
      [{
        name: 'preprocessResult',
        width: this.inputSize[1],
        height: this.inputSize[0],
        // filter: this.gl.NEAREST,
        type: this.gl.FLOAT,
        internalformat: this.gl.RGBA32F,
      }]
    );
  }

  prepareInput(imgSrc) {
    const height = this.inputSize[0];
    const width = this.inputSize[1];

    this.canvas.width = width;
    this.canvas.height = height;
    this.gl_utils.setViewport(
      this.gl.drawingBufferWidth,
      this.gl.drawingBufferHeight
    );

    let imWidth = imgSrc.naturalWidth | imgSrc.videoWidth;
    let imHeight = imgSrc.naturalHeight | imgSrc.videoHeight;
    // assume deeplab_out.width == deeplab_out.height
    let resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    let scaledWidth = Math.floor(imWidth / resizeRatio);
    let scaledHeight = Math.floor(imHeight / resizeRatio);

    this.gl_utils.bindInputTextures(['image']);
    // this.gl_utils.bindTexture('image');
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      imgSrc
    );
    this.preprocessShader.use();
    this.gl_utils.bindFramebuffer('preprocessResult');
    this.gl_utils.render();
    this.gl.readPixels(
      0,
      0,
      width,
      height,
      this.gl.RGB,
      this.gl.FLOAT,
      this.inputTensor
    );

    return [scaledWidth, scaledHeight];
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

  changeModelParam(newModel) {
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelFile = newModel.modelFile;
    this.labelsFile = newModel.labelsFile;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.numClasses = newModel.numClasses;
    this.inputTensor = new Float32Array(newModel.inputSize.reduce((x,y) => x*y));
    this.outputTensor = new Int32Array(newModel.outputSize.reduce((x,y) => x*y));
    this.tfModel = null;
  }
}
