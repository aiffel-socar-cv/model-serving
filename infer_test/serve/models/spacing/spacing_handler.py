from ts.torch_handler.base_handler import BaseHandler


class SpacingHandler(BaseHandler):
    def __init__(self):
        pass

    def initialize(self, context):
        """
        Intialize model.
        This will be called during model loading time
        """

        self.initialized = True

    def preprocess(self, data):
        pass

    def inference(self, model_input):
        pass

    def postprocess(self, inference_output):
        pass

    def handle(self, data, context):
        print("Spacing")
