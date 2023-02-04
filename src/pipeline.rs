use std::path::Path;

use image::imageops::FilterType;
use image::GrayImage;
use tract_onnx::prelude::Tensor;
use tract_onnx::prelude::*;

use super::{GenericTransform, ImageTransform};
use super::{ImageSize, ResizeGrayImage, ToTensor};
use super::{ImageTransformResult, ToArray};

type TractSimplePlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct TransformationPipeline {
    steps: Vec<ImageTransform>,
    model: TractSimplePlan,
}

impl TransformationPipeline {
    pub fn new(image_size: ImageSize) -> Self {
        TransformationPipeline {
            steps: vec![
                ResizeGrayImage {
                    image_size: ImageSize {
                        width: image_size.width,
                        height: image_size.height,
                    },
                    filter: FilterType::CatmullRom,
                }
                .into(),
                ToArray {}.into(),
                ToTensor {}.into(),
            ],
            model: TransformationPipeline::load_model(&image_size),
        }
    }

    pub fn load_model(image_size: &ImageSize) -> TractSimplePlan {
        let name = "common.onnx";
        if !Path::new(name).exists() {
            println!("{name} is not find");
            std::process::exit(-1)
        }
        let input_shape = tvec!(1, 1, image_size.height, image_size.width);
        let mut model = tract_onnx::onnx()
            .model_for_path(name)
            .expect("Cannot read model")
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
            .unwrap();
        model = model.with_output_names(vec!["output"]).unwrap();
        model.into_optimized().unwrap().into_runnable().unwrap()
    }

    pub fn transform_image(&self, image: &GrayImage) -> Result<Tensor, &'static str> {
        let mut result = ImageTransformResult::GrayImage(image.clone());

        for step in &self.steps {
            result = step.transform(result)?;
        }

        let to_tensor = ToTensor {};
        result = to_tensor.transform(result)?;

        match result {
            ImageTransformResult::Tensor(t) => Ok(t),
            _ => Err("Should be converted to tensor already"),
        }
    }

    pub fn extract_features(&self, image: GrayImage) -> Result<Vec<i64>, String> {
        let image_tensor = self
            .transform_image(&image)
            .expect("Cannot transform image");
        let result = self
            .model
            .run(tvec!(image_tensor))
            .expect("Cannot run model");
        let features: Vec<i64> = result[0]
            .to_array_view::<i64>()
            .expect("Cannot extract feature vector")
            .iter()
            .cloned()
            .collect();
        Ok(features)
    }
}
