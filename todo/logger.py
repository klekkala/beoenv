    
    
    
    
    
    
class MyPrintLogger(Logger):
    """Logs results by simply printing out everything."""
    def _init(self):
        # Custom init function.
        print("Initializing ...")
        # Setting up our log-line prefix.
        self.prefix = self.config.get("logger_config").get("prefix")
    def on_result(self, result: dict):
        # Define, what should happen on receiving a `result` (dict).
        print(f"{self.prefix}: {result}")
    def close(self):
        # Releases all resources used by this logger.
        print("Closing")
    def flush(self):
        # Flushing all possible disk writes to permanent storage.
        print("Flushing ;)", flush=True)