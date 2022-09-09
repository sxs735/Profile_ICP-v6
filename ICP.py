import open3d as o3d
Weight_Function = {'None': None,
                   'CauchyLoss' : o3d.pipelines.registration.CauchyLoss(k=0.05),
                   'GMLoss': o3d.pipelines.registration.GMLoss(k=0.05),
                   'TukeyLoss': o3d.pipelines.registration.TukeyLoss(k=0.05),
                   'HuberLoss': o3d.pipelines.registration.HuberLoss(k=0.05),
                   'L1Loss': o3d.pipelines.registration.L1Loss(),
                   'L2Loss': o3d.pipelines.registration.L2Loss()}
ICP_Class = {'PointToPoint' : o3d.pipelines.registration.TransformationEstimationPointToPoint(),
             'PointToPlane' : o3d.pipelines.registration.TransformationEstimationPointToPlane(),
             'Generalized' : o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()}
