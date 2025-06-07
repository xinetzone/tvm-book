# # Load a model
# model = YOLO('yolov5n.yaml')  # build a new model from scratch
# model = YOLO('yolov5n.pt')  # load a pretrained model (recommended for training)
# # Use the model
# # # results = model.train(data='coco8.yaml', epochs=3)  # train the model
# # results = model.train(data='coco.yaml', epochs=3)  # train the model
# # results = model.val()  # evaluate model performance on the validation set
# results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
# results = model.export(format='torchscript')  # "onnx"

# import tvm
# from tvm import relay

# model = torch.jit.load("yolov5nu.torchscript")
# scripted_model = torch.jit.script(model)
# input_name = "x"
# mod, params = relay.frontend.from_pytorch(scripted_model, [(input_name, im.shape)])
# with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
#     lib = relay.build(mod, target="llvm", params=params)
# func = lib[lib.libmod_name]
# module = tvm.contrib.graph_executor.GraphModule(func(tvm.cpu(0)))
# module.run(**{input_name: im.numpy()})
# num_outputs = module.get_num_outputs()
# float_outputs = [module.get_output(k).numpy() for k in range(num_outputs)]
# results = postprocess([torch.from_numpy(o) for o in float_outputs], im, [np.ascontiguousarray(image_np)], self.model.names)
# Image.fromarray(results[0].plot())

