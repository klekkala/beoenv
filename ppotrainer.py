class PPOTorchRLModule(PPORLModule, TorchRLModule):
    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        TorchRLModule.__init__(self, *args, **kwargs)
        PPORLModule.__init__(self, *args, **kwargs)
        #self.blahencoder = VAE(channel_in=3, ch=32)
        #print(self.VAE)
        #checkpoint = torch.load("/lab/kiran/shelL-RL/pretrained_encoder/Models/STL10_ATTARI_84.pt")
        #self.blahencoder.load_state_dict(checkpoint['model_state_dict'])
        
        #self._weights = ResNet18_Weights.IMAGENET1K_V1
        #self.blahencoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        #self._preprocess = self._weights.transforms()
        
        #self.blahencoder.eval()
        #for param in self.encoder.parameters():
        #    print(param.requires_grad)
        
        print(self.encoder)
        print(self.pi)
        print(self.vf)

    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}
        """
        # TODO (Artur): Remove this once Policy supports RNN
        if self.encoder.config.shared:
            batch[STATE_IN] = None
        else:
            batch[STATE_IN] = {
                ACTOR: None,
                CRITIC: None,
            }
        batch[SampleBatch.SEQ_LENS] = None
        

        with torch.no_grad():
            batch['obs'] = self.blahencoder(batch['obs'].cuda()).detach()
            encoder_outs = self.encoder(batch['obs']).detach()

        # TODO (Artur): Un-uncomment once Policy supports RNN
        # output[STATE_OUT] = encoder_outs[STATE_OUT]

        # Actions
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        """
        return output

    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        """PPO forward pass during exploration.
        Besides the action distribution, this method also returns the parameters of the
        policy distribution to be used for computing KL divergence between the old
        policy and the new policy during training.
         """
        with torch.no_grad():
        #    batch['obs'] = self.blahencoder(batch['obs'].cuda())[1]
            return self._common_forward(batch)

    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        #with torch.no_grad():
        #    batch['obs'] = self.blahencoder(batch['obs'].cuda())[1]
        return self._common_forward(batch)

    def _common_forward(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        # TODO (Artur): Remove this once Policy supports RNN

        if self.encoder.config.shared:
            batch[STATE_IN] = None
        else:
            batch[STATE_IN] = {
                ACTOR: None,
                CRITIC: None,
            }
        batch[SampleBatch.SEQ_LENS] = None

        encoder_outs = self.encoder(batch)
        
        # TODO (Artur): Un-uncomment once Policy supports RNN
        # output[STATE_OUT] = encoder_outs[STATE_OUT]

        # Value head
        vf_out = self.vf(encoder_outs[ENCODER_OUT][CRITIC])
        output[SampleBatch.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output