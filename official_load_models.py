from inpainting.ensemble.source_models.Crfill import CrfillAPI
from inpainting.ensemble.source_models.FcF import FcFnetAPI
from inpainting.ensemble.source_models.RFR import RFRNetModelAPI
from inpainting.ensemble.source_models.GMCNN import GMCNNAPI
from inpainting.ensemble.source_models.EdgeCon import ConnectEdgeAPI
from inpainting.ensemble.source_models.Generative import Generative
from inpainting.ensemble.source_models.FcF import FcFnetAPI
from inpainting.ensemble.source_models.Mat import MatAPI

from inpainting.ensemble.source_models.WDnet import WDnet
from inpainting.ensemble.source_models.DBWEModel import DBWRnet
from inpainting.ensemble.source_models.SLBRnet import SLBRnet

def load_models(opt):

    RFRnet=RFRNetModelAPI(opt.dataset,opt)
    GMCNNnet=GMCNNAPI(opt.dataset,opt)
    Crfillnet=CrfillAPI(opt.dataset,opt)
    EdgeConnet=ConnectEdgeAPI(opt)
    Gennet=Generative(dataset=opt.dataset,opt=opt)
    FcFnet=FcFnetAPI(dataset=opt.dataset,device='cuda',opt=opt)
    Matnet=MatAPI(opt)

    ##########watermark remover
    WDModel = WDnet()
    # DBWEModel = DBWRnet()
    SLBRModel = SLBRnet()
    
    RFRnet.eval()
    GMCNNnet.eval()
    Crfillnet.eval()
    EdgeConnet.eval()
    Gennet.eval()
    FcFnet.eval()
    Matnet.eval()
    # return RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet
    # WDModel.eval()
    # DBWEModel.eval()
    # SLBRModel.eval()
    return RFRnet,GMCNNnet,Crfillnet,EdgeConnet,Gennet,FcFnet,Matnet, WDModel, None, SLBRModel
