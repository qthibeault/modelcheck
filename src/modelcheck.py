#!/usr/bin/env python3
#
# Compute the safe region and estimate the robustness of a classifier for a given input
# Copyright (C) 2022 Quinn Thibeault
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math

import click
import matplotlib.pyplot as plt
import numpy as np
import pygad
import rich
import scipy.spatial
import sklearn.decomposition
import torch
import torchvision as vision
import torchvision.transforms.functional as transforms

MODEL_NAMES = ["resnet50", "inceptionV3"]


def _imagenet_classifier(network, transform):
    """Wrap a classifier to support single images and apply a transform before classifying.

    The network output is assumed to be a list of probabilities. The class with maximum probability
    is selected as the class of the image
    
    :param network: The network to use for classification
    :param transform: The image transform to use before classifying
    :type network: torch.nn.Module
    :type transform: callable
    :returns: A function that accepts an image and returns a class
    :rtype: callable
    """

    def classifier(image):
        """Classifier that returns the most-probable class of an image.
        
        :param image: Single-dimensional image
        :type image: torch.Tensor
        :returns: Image class
        :rtype: torch.Tensor
        """

        normalized = transform(image)
        batched = torch.unsqueeze(normalized, 0)
        classes = network(batched)
        return classes.argmax()

    return classifier


def _test_image_class(model, image, expected_class):
    """Ensure the class of the image is the same as the expected class.
    
    :param model: The wrapped model to use for classification
    :param image: The image to classify
    :param expected_class: The class the image should have after classification
    :type model: torch.nn.Module
    :type image: torch.Tensor
    :type expected_class: int
    :raises: RuntimeException
    """

    image_class = model(image)
    if image_class != expected_class:
        raise RuntimeError(f"Classification Error! Expected: {expected_class} -- Found: {image_class}")


def _perturb_image(image, perturbation):
    """Apply a perturbation to an image.

    The perturbation vector is reshaped to have the same dimensions as the image. The image and
    the perturbation are then added together element-wise.

    :param image: The image to perturb
    :param perturbation: The perturbation vector to apply
    :type image: torch.Tensor
    :type perturbation: numpy.ndarray
    :returns: The perturbed image
    :rtype: torch.Tensor
    """

    perturb_tensor = torch.tensor(perturbation, dtype=torch.int16).reshape_as(image)
    return torch.add(image, perturb_tensor)


def _perturbation_fitness(model, original):
    """Create a fitness function given a classifier and an image.

    :param model: The model to use for classification
    :param original: The original input image
    :type model: torch.nn.Module
    :type original: torch.Tensor
    :returns: A function that computes the fitness of a perturbation
    :rtype: callable
    """

    original_class = model(original)

    def fitness_func(perturbation, _index):
        """Compute the fitness of a perturbation.

        The fitness of a perturbation is computed by applying the perturbation to the original
        image and classifying the perturbed image using the model. If the class of the original
        image and the class of the perturbed image is the same, then the fitness is the L2 norm of
        the perturbation vector. Otherwise, the fitness is zero.

        :param perturbation: The perturbation vector to analyze
        :param _index: Unused
        :type perturbation: numpy.ndarray
        :returns: The fitness of the perturbation vector.
        :rtype: int
        """

        perturb_image = _perturb_image(original, perturbation)
        perturb_class = model(perturb_image)

        if perturb_class != original_class:
            return 0

        perturb_scaled = torch.div(torch.tensor(perturbation), 255)
        norm = torch.linalg.vector_norm(perturb_scaled)
        return norm.item()

    return fitness_func


def _step_bounds(pixel_val, step_size):
    """Compute the bounds of a step to update a perturbation.

    :param pixel_val: The pixel value of the original image
    :param step_size: Maximum allowed step size
    :returns: The minimum and maximum allowed step values
    :rtype: (int, int)
    """

    return max(0 - pixel_val, -step_size), min(255 - pixel_val, step_size)


def _mutate_perturbations(original, rng, step_size):
    """Create a mutation function given an image, random number generation, and step size.

    :param original: The original input image
    :param rng: A random number generator
    :param step_size: Size of maximum allowed step
    :type original: torch.Tensor
    :type rng: numpy.random.Generator
    :type step_size: int
    :returns: A mutation function
    :rtype: callable
    """

    pixels = original.flatten()

    def mutation_func(chromosomes, instance):
        """Mutate a population of chromosomes.

        For a given number of mutations, select random indices for each individual in the
        population. At each index, select a random mutation value from the allowed step interval.
        Add the mutation value to the element at the index of the chromosome.

        :param chromosomes: The population of chromosomes to mutate
        :param instance: The instance of the genetic algorithm
        :type chromosomes: numpy.ndarray
        :type instance: pygad.GA
        :returns: The mutated chromosomes
        :rtype: numpy.ndarray
        """

        for chromosome_idx in range(chromosomes.shape[0]):
            chromosome = chromosomes[chromosome_idx]
            mutation_indices = rng.choice(
                range(0, chromosomes.shape[1]),
                size=instance.mutation_num_genes,
                shuffle=False
            )

            for gene_idx in mutation_indices:
                pixel = pixels[gene_idx].item()
                perturbed_pixel = pixel + chromosome[gene_idx]
                lower, upper = _step_bounds(perturbed_pixel, step_size)
                chromosomes[chromosome_idx][gene_idx] += rng.integers(lower, upper, endpoint=True)

        return chromosomes

    return mutation_func


def _create_initial_pop(image, pop_size, rng, step_size):
    """Create a random initial perturbation population.

    :param image: The original input image
    :param pop_size: The size of the population to generate
    :param rng: Random number generator to use for generation
    :param step_size: Maximum allowed perturbation step size
    :type image: torch.Tensor
    :type pop_size: int
    :type rng: numpy.random.Generator
    :type step_size: int
    :returns: A randomly generated set of perturbations
    :rtype: numpy.ndarray
    """

    pixels = image.flatten().numpy()
    bounds = np.array([_step_bounds(pixel, step_size) for pixel in pixels])
    rows = rng.integers(-step_size, step_size, size=(pop_size, pixels.size), dtype=np.short)
    chromosomes = np.clip(rows, bounds[:, 0], bounds[:, 1])

    return np.array(chromosomes)


def _test_model(model, image, pop_size, n_gen, step_size, rng):
    """Search for the safe input region of a model for a given input image.

    To search for the safe input region, this method uses a genetic algorithm to maintain a
    population of perturbations. Tournaments of size 6 are used to select individuals for breeding.
    After a new population is created and evaluated, the top 5% of the prior population replaces the
    bottom 5% of the new population. Once the genetic algorithm is finished, only the individuals
    that still have the same class as the original image are kept, and are converted into perturbed
    images.

    :param model: The model to use for classification
    :param image: The original input image
    :param pop_size: The size of the population to use
    :param n_gen: The number of optimization steps
    :param step_size: The maximum allowed perturbation step size
    :type model: torch.nn.Module
    :type image: torch.Tensor
    :type pop_size: int
    :type n_gen: int
    :type step_size: int
    :returns: A set of images that share the same class as the original image
    :rtype: list
    """

    def print_progress(_alg):
        rich.print(".", end="")

    n_elites = math.floor(0.05 * pop_size)
    initial_population = _create_initial_pop(image, pop_size, rng, step_size)
    fitness_func = _perturbation_fitness(model, image)
    optimizer = pygad.GA(
        num_generations=n_gen,
        on_generation=print_progress,
        fitness_func=fitness_func,
        initial_population=initial_population,
        keep_elitism=n_elites,
        num_parents_mating=2,
        parent_selection_type="tournament",
        K_tournament=6,
        crossover_type="single_point",
        crossover_probability=0.5,
        mutation_type=_mutate_perturbations(image, rng, step_size),
        mutation_percent_genes=5,
        random_seed=rng.integers(0, 2**32 - 1)
    )
    optimizer.run()
    rich.print("done")

    population = [(solution, fitness_func(solution, None)) for solution in optimizer.population]
    images = [(_perturb_image(image, p), f) for p, f in population if f > 0]

    return images


def _ndim_ball_vol(n, radius):
    """Compute the volume of an n-dimensional ball with the given radius.

    :param n: Number of dimensions
    :param radius: Radius of the ball
    :type n: int
    :type radius: float
    :returns: The volume of the ball
    :rtype: float
    """

    return (math.pow(math.pi, n / 2) / scipy.special.gamma(n / 2 + 1)) * math.pow(radius, n)


def _model_robustness(perturbations):
    """Compute the robustness of a given set of perturbed images.

    The robustness metric is computed as the ratio of the volume of convex hull of the perturbed
    images to with the volume of the n-dimensional ball that contains all the images. In order to
    compute the convex hull, it may be necessary to reduce the dimensionality of the perturbed
    images.

    :param perturbations: The set of perturbed images
    :returns: A metric representing the robustness of the set of perturbed images
    :rtype: float
    """

    pca_dim = len(perturbations) - 1
    pca = sklearn.decomposition.PCA(n_components=pca_dim)
    lower_dim_solutions = pca.fit_transform([p.flatten().numpy() for p in perturbations])

    scaled_solutions = [solution / 255 for solution in lower_dim_solutions]
    ball_radius = max(np.linalg.norm(solution) for solution in scaled_solutions)
    ball_volume = _ndim_ball_vol(pca_dim, ball_radius)
    convex_hull = scipy.spatial.ConvexHull(scaled_solutions, qhull_options="Qw Qa Qs")

    return convex_hull.volume / ball_volume


def _show_best(image, save_path):
    """Save or show an image.

    :param image: The image to show
    :param save_path: The path to save the image to
    :type image: torch.Tensor
    :type save_path: str | None
    """

    pil_image = transforms.to_pil_image(image.type(torch.uint8))
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])
    plt.imshow(np.asarray(pil_image))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


@click.command("modelcheck")
@click.option("--model", "model_name", default="resnet50", type=click.Choice(MODEL_NAMES), help="Name of the model to use for testing")
@click.option("--population-size", "pop_size", default=100, help="Size of the population to use for safe-set estimation")
@click.option("--generations", "n_gen", default=50, help="Number of iterations to use for safe-set estimation")
@click.option("--expected-class", default=None, type=click.IntRange(0, 1000), help="The expected class of the input image")
@click.option("--seed", default=None, type=click.IntRange(0, 2**32 - 1), help="Random number generator seed")
@click.option("--step-size", default=5, type=int, help="Size of maximum mutation step")
@click.option("--save-best", "save_path", default=None, type=click.Path(), help="Path to save best image found to")
@click.option("--no-show", "no_show", is_flag=True, help="Do not show maximally perturbed image when finished")
@click.argument("image_path", metavar="IMAGE", type=click.Path(exists=True))
def _model_check(model_name, pop_size, n_gen, expected_class, seed, step_size, save_path, no_show, image_path):
    """Copyright (C) 2022 Quinn Thibeault.

    This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions.
    """

    rich.print("[green]Model:[/green] ", model_name)
    rich.print("[green]Population Size:[/green] ", pop_size)
    rich.print("[green]Generations:[/green] ", n_gen)
    rich.print("[green]Image: [/green]", image_path)

    if expected_class is not None:
        rich.print("[green]Expected Class:[/green] ", expected_class)

    if model_name == "resnet50":
        weights = vision.models.ResNet50_Weights.IMAGENET1K_V2
        network = vision.models.resnet50(weights=weights)
        network.eval()
        model = _imagenet_classifier(network, weights.transforms())
    elif model_name == "inceptionV3":
        weights = vision.models.Inception_V3_Weights.IMAGENET1K_V1
        network = vision.models.inception_v3(weights=weights)
        network.eval()
        model = _imagenet_classifier(network, weights.transforms())
    else:
        raise ValueError(f"Unknown model {model_name}")

    image = vision.io.read_image(click.format_filename(image_path), mode=vision.io.ImageReadMode.RGB)
    rng = np.random.default_rng(seed)
    should_plot = save_path is not None or not no_show

    if expected_class is not None:
        _test_image_class(model, image, expected_class)

    images = _test_model(model, image, pop_size, n_gen, step_size, rng)

    if images:
        robustness = _model_robustness([i[0] for i in images])
        best_image, best_fitness = max(images, key=lambda i: i[1])

        rich.print("[blue]Robustness:[/blue] ", robustness)
        rich.print("[blue]Maximum Perturbation Magnitude:[/blue] ", best_fitness)

        if should_plot:
            _show_best(best_image, save_path)
    else:
        rich.print("[red]No perturbations could be found to compute robustness!![/red]")


if __name__ == "__main__":
    _model_check()
